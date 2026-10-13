#!/usr/bin/env python3
"""MC3 submission: defensive corpus indexing + resident-VLM RAG.

  --index CORPUS   parse every file (catching per file), OCR every image with
                   the VLM, persist a JSON index under INDEX_DIR, and leave the
                   model daemon resident behind a unix socket.
  --query ...      retrieve, ask the resident model for a strict JSON answer,
                   then check it deterministically: ground the value in the
                   text, prune citations to the necessary set, and let a
                   YES/NO verifier veto unsupported answers into refusals.

Local development without a GPU: MC3_SKIP_DAEMON=1 runs the walk and the
plumbing with no model (every answer is a refusal).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path, PurePosixPath

T_START = time.monotonic()

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from retrieval import BM25, chunk_text, doc_status, identifiers, tokenize  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", os.environ.get("MC3_OUTPUT_DIR", "/app/output")))
def _writable_index_dir() -> Path:
    """/app/index normally; a fixed /tmp fallback if the harness runs us as a
    user who cannot write there (index and query must agree, so no randomness)."""
    for cand in (Path(os.environ.get("MC3_INDEX_DIR", "/app/index")), Path("/tmp/mc3-index")):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / f".probe.{os.getpid()}"
            probe.write_text("ok")
            probe.unlink()
            return cand
        except OSError:
            continue
    return Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


INDEX_DIR = _writable_index_dir()
INDEX_FILE = INDEX_DIR / "index.json"
ERRORS_FILE = INDEX_DIR / "errors.json"
IMG_DIR = INDEX_DIR / "img"
SOCK_PATH = Path(os.environ.get("MC3_MODEL_SOCK", str(INDEX_DIR / "model.sock")))
PID_FILE = INDEX_DIR / "daemon.pid"
DAEMON_LOG = INDEX_DIR / "daemon.log"
QUERY_LOG = INDEX_DIR / "queries.log"
DAEMON_SCRIPT = APP_DIR / "daemon.py"
SKIP_DAEMON = os.environ.get("MC3_SKIP_DAEMON", "") == "1"

INDEX_BUDGET_S = float(os.environ.get("MC3_INDEX_BUDGET_S", 560))   # startup budget is 600 s
OCR_STOP_S = float(os.environ.get("MC3_OCR_STOP_S", 500))
QUERY_BUDGET_S = float(os.environ.get("MC3_QUERY_BUDGET_S", 26.5))  # per question is 30 s
PARSE_TIMEOUT_S = 60

# Context size adapts to the GPU: the daemon measures prefill speed at warmup
# and we size the prompt so prefill stays near PREFILL_TARGET_S, within bounds.
MIN_CONTEXT_CHARS = int(os.environ.get("MC3_MIN_CONTEXT_CHARS", 14000))
MAX_CONTEXT_CHARS = int(os.environ.get("MC3_MAX_CONTEXT_CHARS", 56000))
PREFILL_TARGET_S = float(os.environ.get("MC3_PREFILL_TARGET_S", 5.0))
CHARS_PER_TOKEN = 3.2


def context_budget(perf: dict | None) -> int:
    forced = os.environ.get("MC3_CONTEXT_CHARS")
    if forced:
        return int(forced)
    tps = float((perf or {}).get("prefill_tps") or 0)
    if tps <= 0:
        return 26000
    return int(min(MAX_CONTEXT_CHARS, max(MIN_CONTEXT_CHARS, tps * PREFILL_TARGET_S * CHARS_PER_TOKEN)))
QUERY_IMAGE_PIXELS = 900_000
OCR_IMAGE_PIXELS = 1_600_000
MAX_QUERY_IMAGES = 2
JUDGE_MIN_S = float(os.environ.get("MC3_JUDGE_MIN_S", 3.0))  # skip the verifier if less time is left

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
TABULAR_TYPES = {"csv", "tsv", "xlsx", "xlsm", "log"}


def elapsed() -> float:
    return time.monotonic() - T_START


def log(msg: str) -> None:
    print(f"[mc3 {elapsed():6.1f}s] {msg}", file=sys.stderr, flush=True)


# =================================================================== daemon


def _rpc(req: dict, timeout: float) -> dict:
    data = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(max(0.5, timeout))
        s.connect(str(SOCK_PATH))
        s.sendall(data)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    resp = json.loads(buf.decode("utf-8"))
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp


def daemon_ping() -> dict | None:
    if not SOCK_PATH.exists():
        return None
    try:
        r = _rpc({"op": "ping"}, 3.0)
        return r if r.get("ok") else None
    except (OSError, ValueError, RuntimeError):
        return None


def daemon_alive() -> bool:
    return daemon_ping() is not None


def _daemon_process_running() -> bool:
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:  # a zombie still answers kill(0); check its state
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def spawn_daemon() -> None:
    if daemon_alive() or _daemon_process_running():
        return
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(DAEMON_LOG, "ab")
    env = dict(os.environ, MC3_MODEL_SOCK=str(SOCK_PATH))
    proc = subprocess.Popen(
        [sys.executable, str(DAEMON_SCRIPT)],
        stdin=subprocess.DEVNULL,
        stdout=logf,           # never inherit the exec's stdout: docker exec
        stderr=subprocess.STDOUT,  # would wait for the pipe to close
        start_new_session=True,  # must outlive this (and every) app.py process
        close_fds=True,
        cwd=str(APP_DIR),
        env=env,
    )
    PID_FILE.write_text(str(proc.pid))
    log(f"spawned model daemon pid={proc.pid}")


def wait_daemon(deadline: float) -> bool:
    while elapsed() < deadline:
        if daemon_alive():
            return True
        if not _daemon_process_running():
            return False
        time.sleep(0.5)
    return False


# ==================================================================== index


def _walk(corpus: Path) -> tuple[list[tuple[Path, str]], list[dict]]:
    files, errors = [], []

    def onerror(exc: OSError) -> None:
        name = getattr(exc, "filename", "?")
        try:
            rel = Path(name).relative_to(corpus).as_posix()
        except ValueError:
            rel = str(name)
        errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})

    for dirpath, dirnames, filenames in os.walk(corpus, onerror=onerror, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.startswith("~$") or name in ("Thumbs.db", "desktop.ini"):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(corpus).as_posix()
            try:
                if full.is_symlink() or not full.is_file():
                    continue
            except OSError as exc:
                errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
                continue
            files.append((full, rel))
    return files, errors


def _parse_one(full: Path) -> list[dict]:
    from parsers import parse_file

    return parse_file(full)


OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written -- keep every code, "
    "part number, revision, serial number, pin name, value and unit character-for-character. "
    "Keep the reading order and put each line on its own line. If the image is a diagram, "
    "table, pinout or labelled drawing, ALSO list every label together with the item it "
    "belongs to, one pair per line, e.g. 'B14 -> THERM_ALERT#' or 'BOARD REVISION: REV-C2'. "
    "Finish with one line starting 'Description:' saying what the image shows. "
    "No other commentary."
)


def index(corpus: Path) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    corpus = corpus.resolve()

    # Load the model while we parse: the two overlap instead of adding up.
    if not SKIP_DAEMON:
        try:
            spawn_daemon()
        except Exception as exc:
            log(f"daemon spawn failed: {exc}")

    files, errors = _walk(corpus)
    log(f"walk: {len(files)} files, {len(errors)} walk errors")

    docs: list[dict] = []
    ocr_jobs: list[tuple[int, str, str]] = []  # (doc idx, image path, label)
    pool = ThreadPoolExecutor(max_workers=4)
    futures = [(full, rel, pool.submit(_parse_one, full)) for full, rel in files]
    for full, rel, fut in futures:
        ext = full.suffix.lower()
        try:
            units = fut.result(timeout=PARSE_TIMEOUT_S)
        except FutTimeout:
            errors.append({"path": rel, "error": "parse timed out"})
            continue
        except Exception as exc:  # PermissionError, encrypted, corrupt, ...
            errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        texts = [u["text"] for u in units if u["kind"] == "text" and u["text"].strip()]
        imgs = [u for u in units if u["kind"] == "image"]
        if not texts and not imgs:
            errors.append({"path": rel, "error": "no readable content (binary/empty)"})
            continue
        doc = {
            "path": rel,
            "type": ext.lstrip(".") or "file",
            "text": "\n".join(texts),
            "images": [],
            "is_image": ext in IMAGE_EXTS,
            "ocr": [],
        }
        for k, u in enumerate(imgs):
            data = u["image"]
            suffix = ext if (ext in IMAGE_EXTS and k == 0 and u.get("label") == "image") else ".png"
            if suffix == ".png" and not data.startswith(b"\x89PNG"):
                suffix = ".img"
            name = hashlib.sha1(rel.encode() + bytes([k % 256]) + data[:4096]).hexdigest()[:16] + suffix
            p = IMG_DIR / name
            p.write_bytes(data)
            doc["images"].append({"path": str(p), "label": u.get("label", "image")})
            ocr_jobs.append((len(docs), str(p), u.get("label", "image")))
        docs.append(doc)
    pool.shutdown(wait=False, cancel_futures=True)
    log(f"parsed {len(docs)} docs, {len(ocr_jobs)} images to OCR, {len(errors)} unreadable")

    # Save a text-only index right away so queries work even if OCR is cut short.
    _save_index(docs, errors)

    if not SKIP_DAEMON:
        if wait_daemon(INDEX_BUDGET_S - 20):
            log("model daemon is up")
            # standalone pictures first: their content exists nowhere else
            ocr_jobs.sort(key=lambda j: (not docs[j[0]]["is_image"], j[0]))
            for di, img_path, label in ocr_jobs:
                if elapsed() > OCR_STOP_S:
                    log("OCR budget exhausted; remaining images stay query-time only")
                    break
                try:
                    resp = _rpc(
                        {
                            "op": "generate",
                            "messages": [{"role": "user", "content": [
                                {"type": "image", "path": img_path, "max_pixels": OCR_IMAGE_PIXELS},
                                {"type": "text", "text": OCR_PROMPT},
                            ]}],
                            "max_new_tokens": 700,
                            "max_time": 30,
                        },
                        timeout=40,
                    )
                    txt = resp["text"].strip()
                    docs[di]["ocr"].append({"label": label, "text": txt})
                    log(f"OCR {docs[di]['path']} [{label}] {resp.get('secs')}s: {txt[:80]!r}")
                except Exception as exc:
                    log(f"OCR failed for {docs[di]['path']}: {exc}")
            _save_index(docs, errors)
        else:
            log(f"WARNING: model daemon not ready; see {DAEMON_LOG}")
    log("index complete")


def _doc_full_text(doc: dict) -> str:
    parts = []
    if doc["text"].strip():
        parts.append(doc["text"])
    for o in doc.get("ocr", []):
        if doc["is_image"]:
            parts.append(o["text"])
        else:
            parts.append(f"[{o['label']} -- text read from the image]\n{o['text']}")
    return "\n\n".join(parts)


def _save_index(docs: list[dict], errors: list[dict]) -> None:
    full = {d["path"]: _doc_full_text(d) for d in docs}
    status = doc_status(full)
    out_docs, chunks = [], []
    for i, d in enumerate(docs):
        text = full[d["path"]]
        path_words = re.sub(r"[_\-./]", " ", d["path"])
        out_docs.append({
            "path": d["path"], "type": d["type"], "is_image": d["is_image"],
            "text": text, "images": d["images"], "stale": status.get(d["path"], ""),
            "ids": sorted(identifiers(text)),
        })
        pieces = chunk_text(text) or [""]
        for j, piece in enumerate(pieces):
            chunks.append({"doc": i, "n": j, "text": piece,
                           "toks": tokenize(path_words + " " + path_words + "\n" + piece)})
    tmp = INDEX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"docs": out_docs, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, INDEX_FILE)
    ERRORS_FILE.write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")


# ==================================================================== query

SYSTEM_PROMPT = """You are a meticulous analyst answering questions about a company's internal documents. \
You may use ONLY the documents provided. The company and its products are fictional: anything you \
remember from training is wrong by construction and must never be used.

Rules:
1. ANSWER = the value only, copied exactly as the document writes it: a number, code, part number, \
version, pin, revision, date or quarter. Not a sentence. Keep qualifiers that are part of the value \
("Q3 FY27", not "Q3"). Units are optional.
2. The document must state the value for EXACTLY what is asked: the same product/model, the same \
attribute, the same condition (volume tier, revision, environment, date). A value for a different \
product (e.g. TQ-20 instead of TQ-40), a different condition or a merely similar attribute is NOT an \
answer. If the only place the answer could be is missing, unreadable or encrypted, the answer is "".
3. Documents marked STALE (withdrawn, superseded, older revision) must not be used when a current \
document covers the question; the current document wins every conflict. Never cite a STALE document \
unless the question explicitly asks about that old revision.
4. In source code, the value actually assigned to a constant/default wins over comments that mention \
old or previous values.
5. Multi-step questions: when the question points at something in one document (an incident in a log, \
a label, a ticket, a part) and the value lives in another document, follow the exact identifier \
(ticket id, part number, error code, model) from the first document into the second. Never take the \
value from a document that is merely about a similar topic but lacks that identifier.
6. CITATIONS = the minimal set of document paths without which the answer would be impossible: the \
document that states the value, plus a second document only if you needed it to find out WHICH \
identifier/row/item the question is about (and that link is not already given in the question). Do \
not cite documents that only mention the topic, that you used to double-check, or that are STALE.
7. If the documents do not contain the answer, reply with an empty answer and no citations. Do not \
guess, do not approximate, do not compute an answer from unrelated numbers.

Reply with ONE compact JSON object on a single line and nothing else (no markdown):
{"evidence": "<short verbatim quote (max 20 words) of the line that states the answer; for a \
multi-step answer also quote the linking fragment from the other document, separated by ' || '>", \
"answer": "<value or empty string>", "citations": ["<document path>", ...]}"""

JUDGE_SYSTEM = """You are a strict fact checker. Decide whether the documents below explicitly support \
the proposed answer to the question -- the same item, the same attribute and the same conditions the \
question asks about. Documents marked STALE do not count when a current document disagrees. A value \
for a different product, a different condition, or a value the documents do not state is NOT \
supported. Reply with exactly one word: YES or NO."""

REFUSAL_WORDS = re.compile(
    r"^(none|null|n/?a|unknown|not (found|available|specified|stated|provided|given|mentioned|known)|"
    r"no (answer|information|data)|i don'?t know|cannot be determined|not in (the )?(context|documents?))\.?$",
    re.I,
)


def _norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-").replace("‑", "-").replace("−", "-")
    s = s.replace("°", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", _norm(s))


UNIT_SUFFIX_RE = re.compile(
    r"\s*(°\s*[cf]?|degrees?( [cf]| celsius| fahrenheit)?|deg ?c|c|f|k|w|kw|mw|v|mv|a|ma|hz|khz|mhz|ghz|"
    r"s|sec|secs|seconds?|ms|minutes?|mins?|hours?|h|days?|%|percent|gib|gb|mib|mb|kib|kb|tb/s|gb/s|"
    r"tb|mm|cm|m|kg|g|usd|\$|units?|pcs)$",
    re.I,
)


def answer_variants(ans: str) -> list[str]:
    a = _norm(ans).strip(" .;:,'\"`")
    out = {a}
    core = UNIT_SUFFIX_RE.sub("", a).strip()
    if core:
        out.add(core)
    core2 = re.sub(r"^(pin|version|v|rev(ision)?|firmware|meridian|part( number)?|p/n|code|error|quarter)\s+", "", core)
    if core2:
        out.add(core2)
    if re.fullmatch(r"[$]?[\d,]+(\.\d+)?", core2 or ""):
        out.add(core2.replace(",", "").lstrip("$"))
    return [v for v in out if v]


def contains_answer(text: str, ans: str) -> bool:
    if not ans.strip():
        return False
    nt = _norm(text)
    ct = _compact(text)
    nt_nocomma = re.sub(r"(?<=\d),(?=\d{3})", "", nt)
    for v in answer_variants(ans):
        if re.fullmatch(r"[\d.]+", v):
            if re.search(rf"(?<![\d.]){re.escape(v)}(?![\d]|\.\d)", nt_nocomma):
                return True
            continue
        if len(v) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", nt):
            return True
        cv = _compact(v)
        if len(cv) >= 4 and cv in ct:
            return True
    return False


NUM_UNIT_RE = re.compile(
    r"^([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:°\s*[CF]?|degrees?(?:\s*(?:C|F|celsius|fahrenheit))?|deg\s*C|[CFK]|"
    r"W|kW|mW|V|mV|A|mA|Hz|kHz|MHz|GHz|s|secs?|seconds?|ms|minutes?|mins?|hours?|hrs?|days?|%|percent|"
    r"GiB|GB|MiB|MB|KiB|KB|TiB|TB|TB/s|GB/s|Gbps|Mbps|mm|cm|m|kg|g|USD|units?|pcs|RPM|rpm)\.?$",
    re.I,
)


def _clean_answer(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (int, float)):
        raw = str(int(raw)) if isinstance(raw, float) and raw.is_integer() else str(raw)
    if isinstance(raw, list):
        raw = ", ".join(str(x) for x in raw)
    a = str(raw).strip().strip("`").strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'":
        a = a[1:-1].strip()
    a = a.rstrip(".").strip()
    if REFUSAL_WORDS.match(a):
        return ""
    # "94 C" / "31 W" / "180 seconds" -> the bare value (the grader normalises
    # units anyway; the bare number is the canonical expected form)
    m = NUM_UNIT_RE.match(a)
    if m:
        a = m.group(1)
    return a


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        return {}
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    break
    # truncated / slightly broken JSON: pull the fields out by regex
    obj = {}
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        obj["answer"] = m.group(1)
    m = re.search(r'"citations"\s*:\s*\[([^\]]*)\]', text)
    if m:
        obj["citations"] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    m = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if m:
        obj["evidence"] = m.group(1)
    return obj


class Retriever:
    def __init__(self, index: dict):
        self.docs = index["docs"]
        self.chunks = index["chunks"]
        self.bm25 = BM25([c["toks"] for c in self.chunks])
        self.by_doc: dict[int, list[int]] = {}
        for ci, c in enumerate(self.chunks):
            self.by_doc.setdefault(c["doc"], []).append(ci)
        self.path_to_doc = {d["path"]: i for i, d in enumerate(self.docs)}

    def doc_scores(self, chunk_scores: list[float]) -> dict[int, float]:
        best: dict[int, list[float]] = {}
        for ci, s in enumerate(chunk_scores):
            best.setdefault(self.chunks[ci]["doc"], []).append(s)
        out = {}
        for di, ss in best.items():
            ss.sort(reverse=True)
            s = ss[0] + 0.15 * sum(ss[1:3])
            if self.docs[di]["stale"]:
                s *= 0.8
            out[di] = s
        return out

    def select(self, question: str, budget: int = 26000) -> tuple[list[tuple[int, list[int] | None]], dict[int, float]]:
        """Return ([(doc index, chunk indices or None for whole doc)], doc scores) in rank order."""
        qt = tokenize(question)
        cs = self.bm25.scores(qt) if self.chunks else []
        ds = self.doc_scores(cs)
        ranked = sorted(range(len(self.docs)), key=lambda d: -ds.get(d, 0.0))
        total = sum(len(d["text"]) + 150 for d in self.docs)
        if total <= budget:
            return [(d, None) for d in ranked], ds

        # Large corpus: top chunks first, then pull in chunks that share a
        # distinctive identifier with the best hits (the multi-hop bridge).
        order = sorted(range(len(cs)), key=lambda i: -cs[i])
        picked: list[int] = []
        used = 0
        for ci in order[:40]:
            if cs[ci] <= 0 or used > budget * 0.6:
                break
            picked.append(ci)
            used += len(self.chunks[ci]["text"]) + 40
        q_ids = identifiers(question)
        bridge_ids: list[str] = []
        for ci in picked[:5]:
            for x in identifiers(self.chunks[ci]["text"]):
                if x not in q_ids and x not in bridge_ids:
                    bridge_ids.append(x)
        q_ids_l = [x for x in q_ids]
        extra = []
        for ci, c in enumerate(self.chunks):
            if ci in picked:
                continue
            up = c["text"].upper()
            hit_q = sum(1 for x in q_ids_l if x.upper() in up)
            hit_b = sum(1 for x in bridge_ids if x.upper() in up)
            if hit_q or hit_b:
                extra.append((2.0 * hit_q + hit_b + 0.01 * cs[ci], ci))
        extra.sort(reverse=True)
        for _, ci in extra:
            if used > budget:
                break
            picked.append(ci)
            used += len(self.chunks[ci]["text"]) + 40
        # keep standalone images that rank at all: they are short and often decisive
        per_doc: dict[int, list[int]] = {}
        for ci in picked:
            per_doc.setdefault(self.chunks[ci]["doc"], []).append(ci)
        sel = sorted(per_doc, key=lambda d: -ds.get(d, 0.0))
        return [(d, sorted(per_doc[d])) for d in sel], ds


def _doc_block(doc: dict, chunk_ids: list[int] | None, chunks: list[dict], attached: dict[str, int]) -> str:
    attrs = [f'path="{doc["path"]}"', f'type="{doc["type"]}"']
    if doc["stale"]:
        attrs.append(f'status="STALE: {doc["stale"]}"')
    if doc["is_image"]:
        note = "text transcribed from the image"
        if doc["path"] in attached:
            note += f"; the image itself is attached as Picture {attached[doc['path']]}"
        attrs.append(f'note="{note}"')
    if chunk_ids is None:
        body = doc["text"]
    else:
        parts, last = [], None
        for ci in chunk_ids:
            if last is not None and chunks[ci]["n"] != last + 1:
                parts.append("[...]")
            parts.append(chunks[ci]["text"])
            last = chunks[ci]["n"]
        body = "\n".join(parts)
    return f"<document {' '.join(attrs)}>\n{body.strip()}\n</document>"


IMAGE_HINT_RE = re.compile(
    r"\b(image|picture|photo|label|sticker|diagram|drawing|figure|pinout|pin|schematic|screenshot|"
    r"chart|graph|plot|scan|printed|shown|depicted|silkscreen|nameplate|badge|map|layout)\b",
    re.I,
)


def _quick_index(corpus: Path) -> None:
    """--index never ran (or its output is gone): parse the corpus text-only,
    right now, and make sure the daemon is coming up for later questions."""
    log("no index on disk -- building a text-only index inside the query")
    global OCR_STOP_S
    OCR_STOP_S = -1  # no OCR inside a question's budget
    files, errors = _walk(corpus.resolve())
    docs = []
    for full, rel in files:
        try:
            units = _parse_one(full)
        except Exception as exc:
            errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        text = "\n".join(u["text"] for u in units if u["kind"] == "text")
        if text.strip():
            docs.append({"path": rel, "type": full.suffix.lower().lstrip(".") or "file", "text": text,
                         "images": [], "is_image": False, "ocr": []})
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _save_index(docs, errors)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    if not INDEX_FILE.exists():
        _quick_index(corpus)
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    R = Retriever(index_data)
    docs = R.docs
    if not docs:
        return "", [], 0.0
    perf = None
    if not SKIP_DAEMON:
        perf = daemon_ping()
        if perf is None:
            spawn_daemon()
            if not wait_daemon(QUERY_BUDGET_S - 4):
                _qlog({"query": query, "error": "daemon unavailable"})
                return "", [], 0.0
            perf = daemon_ping()
    budget = context_budget(perf)
    selection, dscores = R.select(query, budget)
    rank = {d: r for r, (d, _) in enumerate(selection)}

    # Images to attach: standalone pictures (or rendered pages) of top-ranked docs.
    attach: list[tuple[str, str]] = []  # (doc path, image file)
    image_question = bool(IMAGE_HINT_RE.search(query))
    for d, _ in selection[:4]:
        doc = docs[d]
        if not doc["images"] or dscores.get(d, 0) <= 0:
            continue
        if not (doc["is_image"] or image_question or not doc["text"].strip()):
            continue
        if rank[d] >= (3 if image_question else 2):
            continue
        for im in doc["images"][:1]:
            if os.path.exists(im["path"]) and len(attach) < MAX_QUERY_IMAGES:
                attach.append((doc["path"], im["path"]))
    attached = {p: i + 1 for i, (p, _) in enumerate(attach)}

    blocks = [_doc_block(docs[d], cids, R.chunks, attached) for d, cids in selection]
    user_text = (
        f"QUESTION: {query}\n\nDOCUMENTS:\n\n" + "\n\n".join(blocks)
        + f"\n\nQUESTION (repeated): {query}\n"
        "Answer from the documents only, following every rule. Output the JSON object only."
    )
    content = []
    for _, img in attach:
        content.append({"type": "image", "path": img, "max_pixels": QUERY_IMAGE_PIXELS})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    if SKIP_DAEMON:
        return "", [], 0.0

    remaining = QUERY_BUDGET_S - elapsed()
    resp = _rpc({"op": "generate", "messages": messages, "max_new_tokens": 200,
                 "prefix": '{"evidence": "', "max_time": max(3.0, remaining - 2.5)}, timeout=remaining)
    raw = resp["text"]
    obj = parse_json_object(raw)
    ans = _clean_answer(obj.get("answer"))
    trace = {"query": query, "raw": raw, "gen_secs": resp.get("secs"), "in_tokens": resp.get("in_tokens"),
             "attached": [p for p, _ in attach], "budget": budget, "perf": perf, "context_docs": [docs[d]["path"] for d, _ in selection]}
    if not ans:
        _qlog({**trace, "final": ["", []]})
        return "", [], 0.0

    shown = {docs[d]["path"] for d, _ in selection}
    cites = _resolve_citations(obj.get("citations"), shown)
    evidence = str(obj.get("evidence") or "")
    cites = _finalize_citations(query, ans, cites, evidence, R, selection, attached)
    trace["citations_after_rules"] = cites

    grounded = any(contains_answer(docs[R.path_to_doc[c]]["text"], ans) for c in cites)
    grounded_anywhere = grounded or any(contains_answer(d["text"], ans) for d in docs)
    used_image = any(c in attached for c in cites)
    if not cites:
        _qlog({**trace, "final": ["", []], "why": "no citation survived"})
        return "", [], 0.0

    # A numeric condition in the question ("at 10,000 unit volume") must be
    # visible in the evidence; otherwise the value answers a different question.
    cited_text = "\n".join(docs[R.path_to_doc[c]]["text"] for c in cites)
    missing = [n for n in condition_numbers(query) if not mentions_number(cited_text, n)]
    if missing and not used_image:
        _qlog({**trace, "final": ["", []], "why": f"question condition(s) {missing} absent from cited evidence"})
        return "", [], 0.0

    # Verifier: a second, independent read that can veto into a refusal.
    p_yes = 0.75
    remaining = QUERY_BUDGET_S - elapsed()
    if remaining > JUDGE_MIN_S:
        try:
            p_yes = _judge(query, ans, cites, R, attach, remaining - 1.0)
        except Exception as exc:
            trace["judge_error"] = str(exc)
    trace["p_yes"] = p_yes
    threshold = 0.10 if (grounded or used_image) else 0.5
    if not grounded_anywhere and not used_image:
        threshold = 0.6
    if p_yes < threshold:
        _qlog({**trace, "final": ["", []], "why": f"verifier veto p_yes={p_yes:.2f}"})
        return "", [], 0.0
    conf = round(min(0.99, 0.5 * p_yes + (0.4 if grounded else 0.2)), 3)
    _qlog({**trace, "final": [ans, cites], "grounded": grounded})
    return ans, cites, conf


COND_NUM_RE = re.compile(r"(?<![\w.,-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*[kKmM]\b)?(?![\w.-]*\d)(?!-)")


def condition_numbers(question: str) -> list[str]:
    """Numbers the question states as a CONDITION ("at 10,000 unit volume",
    "at 50% load"): bare numbers that are not part of an identifier (TQ-40,
    ORR-1847, FY27, 4.3.2) and not years."""
    out = []
    for m in COND_NUM_RE.finditer(question):
        start = m.start()
        if start > 0 and (question[start - 1].isalnum() or question[start - 1] in "-_/#:"):
            continue
        if question[m.end():m.end() + 1] == ":":
            continue  # clock times are not conditions
        num = m.group(1).replace(",", "")
        if re.fullmatch(r"(19|20)\d\d", num) or len(num) < 2:
            continue
        mult = (m.group(2) or "").strip().lower()
        if mult == "k":
            num = str(int(float(num) * 1000))
        elif mult == "m":
            num = str(int(float(num) * 1_000_000))
        out.append(num)
    return out


def mentions_number(text: str, num: str) -> bool:
    t = re.sub(r"(?<=\d)[,\u202f\u2009 ](?=\d{3}(?!\d))", "", text)
    if re.search(rf"(?<![\d.]){re.escape(num)}(?![\d]|\.\d)", t):
        return True
    n = int(float(num)) if re.fullmatch(r"\d+(\.\d+)?", num) else None
    if n and n >= 1000 and n % 1000 == 0:
        k = n // 1000
        if re.search(rf"(?<![\d.]){k}\s*[kK]\b", t):
            return True
        if n % 1_000_000 == 0 and re.search(rf"(?<![\d.]){n // 1_000_000}\s*[mM]\b", t):
            return True
    return False


def _resolve_citations(raw, shown: set[str]) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for c in raw or []:
        c = str(c).strip().strip("/").replace("\\", "/")
        if c.startswith("app/corpus/"):
            c = c[len("app/corpus/"):]
        if c in shown:
            hit = c
        else:  # tolerate a bare file name or a partial path
            cands = [p for p in shown if p.endswith("/" + c) or PurePosixPath(p).name == PurePosixPath(c).name]
            hit = cands[0] if len(cands) == 1 else None
        if hit and hit not in out:
            out.append(hit)
    return out


def _answer_lines(text: str, ans: str) -> list[str]:
    return [ln for ln in text.splitlines() if contains_answer(ln, ans)]


def _finalize_citations(query, ans, cites, evidence, R, selection, attached) -> list[str]:
    docs = R.docs
    shown = [docs[d]["path"] for d, _ in selection]
    D = lambda p: docs[R.path_to_doc[p]]  # noqa: E731

    # Documents whose text the model quoted as evidence are candidates too.
    if evidence:
        segs = re.findall(r"\"([^\"]{18,})\"|“([^”]{18,})”", evidence)
        segs = [a or b for a, b in segs]
        segs += re.split(r"\n|;\s+|\s+\|\|\s+|\.\.\.|…", evidence)
        for seg in segs:
            seg = _norm(re.sub(r"^\s*[\w./-]{1,40}:\s+(?=\S)", "", seg).strip(" \"'“”"))
            if len(seg) < 18:
                continue
            probes = {seg[:60], seg[-60:]}
            for p in shown:
                if p in cites:
                    continue
                t = _norm(D(p)["text"])
                if any(len(pr) >= 18 and pr in t for pr in probes):
                    cites.append(p)

    has_answer = {p: contains_answer(D(p)["text"], ans) for p in shown}
    current_with_answer = [p for p in shown if has_answer[p] and not D(p)["stale"]]
    q_low = query.lower()
    asks_old = bool(re.search(r"\b(withdrawn|superseded|old(er)?|previous|original|first|r1|rev(ision)? ?1)\b", q_low))

    # 1. stale documents go when a current document carries the answer
    if not asks_old and current_with_answer:
        cites = [c for c in cites if not D(c)["stale"]]

    ans_docs = [c for c in cites if has_answer[c]]
    if not ans_docs:
        # model cited only a bridge (or nothing grounded): add the unique current answer holder
        if len(current_with_answer) == 1 and not any(c in attached for c in cites):
            cites = cites + current_with_answer
            ans_docs = current_with_answer
        elif not cites and len(current_with_answer) >= 1:
            best = min(current_with_answer, key=lambda p: shown.index(p))
            cites, ans_docs = [best], [best]

    if not ans_docs:
        # ungrounded in text (typically an image read directly): trust the model, minus stale
        keep = [c for c in cites if not D(c)["stale"]] or cites
        return keep[:2]

    # 2. answer holders: if several, keep the best-ranked ones the model chose (max 2)
    if len(ans_docs) > 2:
        ans_docs = sorted(ans_docs, key=lambda p: shown.index(p))[:2]

    # 3. non-answer citations survive only as genuine bridges
    q_ids = {x.upper() for x in identifiers(query)}
    a_ids = {x.upper() for x in identifiers(ans)} | {_norm(ans).upper()}
    final = list(ans_docs)
    for c in cites:
        if c in final or D(c)["stale"] and not asks_old:
            continue
        c_ids = {x.upper() for x in D(c)["ids"]}
        bridged = False
        for a in ans_docs:
            atext = D(a)["text"]
            lines = _answer_lines(atext, ans)
            near = set()
            for ln in lines:
                near |= {x.upper() for x in identifiers(ln)}
            if near & q_ids:
                continue  # the question's own identifier already locates the answer row
            near = set(near)
            near -= a_ids
            if not near or D(a)["type"] not in TABULAR_TYPES and not near - q_ids:
                near = {x.upper() for x in D(a)["ids"]}
            link = (near & c_ids) - q_ids - a_ids
            if link:
                bridged = True
                break
        if bridged:
            final.append(c)
    return final


JUDGE_DOC_CHARS = 3500


def _excerpt(text: str, ans: str, terms: set[str]) -> str:
    """Short docs whole; long docs as the answer lines, the lines carrying a
    linking identifier, their neighbours and the opening lines."""
    if len(text) <= JUDGE_DOC_CHARS:
        return text
    lines = text.splitlines()
    keep = set(range(min(4, len(lines))))
    for i, ln in enumerate(lines):
        up = ln.upper()
        if contains_answer(ln, ans) or any(t in up for t in terms):
            keep.update(range(max(0, i - 1), min(len(lines), i + 2)))
    out, last = [], -1
    for i in sorted(keep):
        if i != last + 1:
            out.append("[...]")
        out.append(lines[i])
        last = i
    return "\n".join(out)[: JUDGE_DOC_CHARS * 2]


def _judge(query, ans, cites, R, attach, budget) -> float:
    docs = R.docs
    cited_docs = [docs[R.path_to_doc[c]] for c in cites]
    # identifiers shared between cited documents are the multi-hop links
    terms = {x.upper() for x in identifiers(query)}
    if len(cited_docs) > 1:
        id_sets = [set(d["ids"]) for d in cited_docs]
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                terms |= {x.upper() for x in id_sets[i] & id_sets[j]}
    blocks = []
    for c, d in zip(cites, cited_docs):
        text = _excerpt(d["text"], ans, terms)
        st = f' status="STALE: {d["stale"]}"' if d["stale"] else ""
        blocks.append(f'<document path="{c}"{st}>\n{text.strip()}\n</document>')
    content = []
    for p, img in attach:
        if p in cites:
            content.append({"type": "image", "path": img, "max_pixels": QUERY_IMAGE_PIXELS})
    content.append({"type": "text", "text": (
        "DOCUMENTS:\n\n" + "\n\n".join(blocks)
        + f"\n\nQUESTION: {query}\nPROPOSED ANSWER: {ans}\n\n"
        "Do these documents explicitly support exactly this answer to exactly this question? "
        "Answer YES or NO."
    )})
    resp = _rpc({"op": "choose", "choices": ["YES", "NO"],
                 "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": content}]}, timeout=budget)
    return float(resp["probs"].get("YES", 0.5))


def _qlog(rec: dict) -> None:
    try:
        rec["t"] = round(elapsed(), 2)
        with open(QUERY_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ================================================================= plumbing


def _write_output(query_id: str, text: str, citations: list[str], confidence: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (query_id + "_output.json")
    tmp = OUTPUT_DIR / (f".{query_id}_output.json.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"answer": text, "citations": list(citations), "confidence": float(confidence)},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, out)  # atomic: the harness never sees a half-written file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # do not wait on a parser thread that may be stuck on a hostile file

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    # A well-formed refusal goes down first: whatever happens next, the file
    # the harness reads is valid JSON with both graded keys.
    _write_output(args.query_id, "", [], 0.0)
    try:
        text, citations, confidence = answer(args.corpus, args.query)
    except Exception as exc:
        log(f"answer failed: {type(exc).__name__}: {exc}")
        _qlog({"query": args.query, "error": f"{type(exc).__name__}: {exc}"})
        return 0
    _write_output(args.query_id, text, citations, confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
