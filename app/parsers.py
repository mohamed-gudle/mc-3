"""File parsing for the MC3 corpus.

Every parser returns a list of "units". Each unit is a dict:

    {"kind": "text",  "text": str}                 -- extracted text
    {"kind": "image", "image": bytes, "label": str} -- an image to OCR with the VLM
                                                       (a standalone picture, an
                                                       image embedded in a docx/
                                                       pptx, or a rendered PDF page)

Parsers RAISE on unreadable / encrypted content. The walker in index.py catches
per file and keeps going -- one bad file must never cost the rest of the corpus.

Deliberately dependency-light: docx/pptx/xlsx are parsed from their XML with
the standard library (openpyxl is used for xlsx when present, for cached
formula values), PDFs use PyMuPDF when present and pypdf otherwise.
"""

from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx", ".docm", ".dotx"}
PPTX_EXTS = {".pptx", ".pptm"}
XLSX_EXTS = {".xlsx", ".xlsm"}
CSV_EXTS = {".csv", ".tsv"}
HTML_EXTS = {".html", ".htm", ".xhtml"}

MAX_TEXT_BYTES = 8 << 20          # never slurp more than 8 MiB of text from one file
MAX_IMAGE_BYTES = 40 << 20
MIN_EMBEDDED_IMAGE_BYTES = 2048   # skip bullets / spacer gifs inside office docs
PDF_RENDER_DPI = 150
PDF_TEXT_SPARSE_CHARS = 200       # a page with less text than this gets OCR'd
PDF_MAX_RENDERED_PAGES = 12       # per file


class Unreadable(Exception):
    """Content exists but cannot be read (encrypted, corrupt, ...)."""


# ------------------------------------------------------------------ helpers


def looks_binary(raw: bytes, threshold: float = 0.05) -> bool:
    """Decide on the RAW bytes: NULs are decisive, else share of control bytes."""
    head = raw[:16384]
    if not head:
        return False
    if b"\x00" in head:
        # UTF-16 text has NULs too; accept it only with a BOM.
        return not (head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"))
    bad = sum(1 for b in head if b < 9 or (13 < b < 32) or b == 127)
    return bad / len(head) > threshold


def decode_text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def mostly_garbage(text: str, threshold: float = 0.15) -> bool:
    if not text:
        return False
    sample = text[:20000]
    bad = sum(1 for ch in sample if ch == "�" or (not ch.isprintable() and ch not in "\n\t\r"))
    return bad / len(sample) > threshold


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return repr(v)
    return str(v).strip()


def render_table(rows: list[list[str]], title: str = "") -> str:
    """Render a table so EVERY row is self-contained: 'col: val | col: val'.

    Rows that end up in different retrieval chunks still carry their header,
    which is what lets a question about one row find it and read it right.
    """
    rows = [[_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    out = []
    if title:
        out.append(f"## {title}")
    header = rows[0]
    has_header = len(rows) > 1 and all(header) and not any(_numeric(c) for c in header)
    if has_header:
        out.append("columns: " + " | ".join(header))
        for r in rows[1:]:
            parts = []
            for i, c in enumerate(r):
                if not c:
                    continue
                name = header[i] if i < len(header) and header[i] else f"col{i + 1}"
                parts.append(f"{name}: {c}")
            out.append(" | ".join(parts))
    else:
        for r in rows:
            out.append(" | ".join(c for c in r if c))
    return "\n".join(out)


def _numeric(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?[\d.,]+%?", s.strip()))


# ------------------------------------------------------------------ parsers


def parse_plain(path: Path) -> list[dict]:
    raw = path.read_bytes()[:MAX_TEXT_BYTES]
    if looks_binary(raw):
        return []
    text = decode_text(raw)
    if mostly_garbage(text):
        return []
    if path.suffix.lower() in HTML_EXTS:
        text = html_to_text(text)
    return [{"kind": "text", "text": text}]


def html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d|table)>", "\n", text)
    text = re.sub(r"(?i)</t[dh]>", " | ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def parse_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()[:MAX_TEXT_BYTES]
    if looks_binary(raw):
        return []
    text = decode_text(raw)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    rendered = render_table(rows)
    return [{"kind": "text", "text": rendered or text}]


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _zip_open(path: Path) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        # Encrypted OOXML is an OLE compound file, not a zip.
        raise Unreadable(f"not a readable OOXML package ({exc})") from exc
    return zf


def _docx_block_text(el) -> str:
    parts = []
    for node in el.iter():
        tag = node.tag
        if tag == W_NS + "t" and node.text:
            parts.append(node.text)
        elif tag == W_NS + "tab":
            parts.append("\t")
        elif tag in (W_NS + "br", W_NS + "cr"):
            parts.append("\n")
    return "".join(parts)


def _docx_xml_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    body = root.find(W_NS + "body")
    container = body if body is not None else root
    out: list[str] = []

    def walk(parent):
        for child in parent:
            if child.tag == W_NS + "p":
                t = _docx_block_text(child).strip()
                if t:
                    out.append(t)
            elif child.tag == W_NS + "tbl":
                rows = []
                for tr in child.iter(W_NS + "tr"):
                    rows.append([_docx_block_text(tc).strip() for tc in tr.findall(W_NS + "tc")])
                t = render_table(rows)
                if t:
                    out.append(t)
            elif child.tag == W_NS + "sdt":
                content = child.find(W_NS + "sdtContent")
                if content is not None:
                    walk(content)

    walk(container)
    return "\n".join(out)


def _embedded_images(zf: zipfile.ZipFile, prefix: str) -> list[dict]:
    units = []
    for name in sorted(zf.namelist()):
        if not name.startswith(prefix):
            continue
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        info = zf.getinfo(name)
        if info.file_size < MIN_EMBEDDED_IMAGE_BYTES or info.file_size > MAX_IMAGE_BYTES:
            continue
        units.append({"kind": "image", "image": zf.read(name), "label": f"embedded image {Path(name).name}"})
    return units


def parse_docx(path: Path) -> list[dict]:
    with _zip_open(path) as zf:
        names = set(zf.namelist())
        if "word/document.xml" not in names:
            raise Unreadable("docx without word/document.xml")
        texts = [_docx_xml_text(zf.read("word/document.xml"))]
        for extra in sorted(names):
            if re.match(r"word/(header|footer|footnotes|endnotes)\d*\.xml$", extra):
                try:
                    t = _docx_xml_text(zf.read(extra)).strip()
                except ET.ParseError:
                    continue
                if t:
                    texts.append(t)
        units = [{"kind": "text", "text": "\n".join(texts)}]
        units += _embedded_images(zf, "word/media/")
    return units


def parse_pptx(path: Path) -> list[dict]:
    with _zip_open(path) as zf:
        slides = sorted(
            (n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
        )
        out = []
        for i, name in enumerate(slides, 1):
            root = ET.fromstring(zf.read(name))
            paras = []
            for p in root.iter(A_NS + "p"):
                t = "".join(x.text or "" for x in p.iter(A_NS + "t")).strip()
                if t:
                    paras.append(t)
            notes_name = f"ppt/notesSlides/notesSlide{i}.xml"
            if notes_name in zf.namelist():
                nroot = ET.fromstring(zf.read(notes_name))
                notes = " ".join(x.text or "" for x in nroot.iter(A_NS + "t")).strip()
                if notes:
                    paras.append(f"(speaker notes) {notes}")
            out.append(f"## Slide {i}\n" + "\n".join(paras))
        units = [{"kind": "text", "text": "\n\n".join(out)}]
        units += _embedded_images(zf, "ppt/media/")
    return units


def parse_xlsx(path: Path) -> list[dict]:
    try:
        import openpyxl
    except ImportError:  # pragma: no cover - fallback when openpyxl is absent
        return _parse_xlsx_raw(path)
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except zipfile.BadZipFile as exc:
        raise Unreadable(f"not a readable xlsx ({exc})") from exc
    out = []
    try:
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            t = render_table(rows, title=f"sheet: {ws.title}")
            if t:
                out.append(t)
    finally:
        wb.close()
    return [{"kind": "text", "text": "\n\n".join(out)}]


def _parse_xlsx_raw(path: Path) -> list[dict]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with _zip_open(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
        out = []
        for name in sorted(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
            rows = []
            for row in ET.fromstring(zf.read(name)).iter(ns + "row"):
                vals = []
                for c in row.iter(ns + "c"):
                    v = c.find(ns + "v")
                    if c.get("t") == "s" and v is not None:
                        vals.append(shared[int(v.text)])
                    elif c.get("t") == "inlineStr":
                        vals.append("".join(t.text or "" for t in c.iter(ns + "t")))
                    else:
                        vals.append(v.text if v is not None else "")
                rows.append(vals)
            out.append(render_table(rows, title=Path(name).stem))
    return [{"kind": "text", "text": "\n\n".join(out)}]


def parse_pdf(path: Path) -> list[dict]:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # older PyMuPDF
    except ImportError:
        return _parse_pdf_pypdf(path)
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise Unreadable(f"pdf open failed: {exc}") from exc
    with doc:
        # The contract: an encrypted PDF is unreadable. Do NOT try empty or
        # guessed passwords -- an answer that only exists in an encrypted file
        # must be refused, not cracked.
        if doc.is_encrypted or doc.needs_pass:
            raise Unreadable("encrypted pdf")
        pages_text = []
        images = []
        for pno, page in enumerate(doc):
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception:
                text = ""
            pages_text.append(f"[page {pno + 1}]\n{text.strip()}")
            if len(images) >= PDF_MAX_RENDERED_PAGES:
                continue
            sparse = len(text.strip()) < PDF_TEXT_SPARSE_CHARS
            img_heavy = False
            try:
                area = abs(page.rect)
                covered = 0.0
                for info in page.get_image_info():
                    bbox = fitz.Rect(info.get("bbox", (0, 0, 0, 0)))
                    covered += abs(bbox & page.rect)
                img_heavy = area > 0 and covered / area > 0.12
            except Exception:
                pass
            if sparse or img_heavy:
                try:
                    pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
                    images.append({"kind": "image", "image": pix.tobytes("png"),
                                   "label": f"page {pno + 1} rendering"})
                except Exception:
                    pass
    return [{"kind": "text", "text": "\n\n".join(pages_text)}] + images


def _parse_pdf_pypdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise Unreadable(f"pdf open failed: {exc}") from exc
    if reader.is_encrypted:
        raise Unreadable("encrypted pdf")
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"[page {i + 1}]\n{(page.extract_text() or '').strip()}")
        except Exception:
            continue
    return [{"kind": "text", "text": "\n\n".join(pages)}]


def parse_image(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise Unreadable("image too large")
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:  # validate it really is an image
        im.verify()
    return [{"kind": "image", "image": raw, "label": "image"}]


def sniff_parse(path: Path) -> list[dict]:
    """Unknown extension: dispatch on magic bytes, else treat as text if it is text."""
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head.startswith(b"%PDF"):
        return parse_pdf(path)
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or head[:4] in (b"GIF8", b"RIFF"):
        try:
            return parse_image(path)
        except Exception:
            return []
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return []
        if "word/document.xml" in names:
            return parse_docx(path)
        if "xl/workbook.xml" in names:
            return parse_xlsx(path)
        if any(n.startswith("ppt/") for n in names):
            return parse_pptx(path)
        return []
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise Unreadable("legacy/encrypted OLE office file")
    return parse_plain(path)


def parse_file(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return parse_image(path)
    if ext in PDF_EXTS:
        return parse_pdf(path)
    if ext in DOCX_EXTS:
        return parse_docx(path)
    if ext in PPTX_EXTS:
        return parse_pptx(path)
    if ext in XLSX_EXTS:
        return parse_xlsx(path)
    if ext in CSV_EXTS:
        return parse_csv(path)
    return sniff_parse(path)
