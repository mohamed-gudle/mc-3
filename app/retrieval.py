"""Chunking, lexical retrieval, identifier linking and document status.

Pure standard library: the per-question process imports this and nothing
heavy, so a question costs a socket round-trip plus a few milliseconds here.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

# ----------------------------------------------------------------- tokens

WORD_RE = re.compile(r"[a-z0-9]+")
COMPOUND_RE = re.compile(r"[a-z0-9]+(?:[-_./:#][a-z0-9]+)+")
STOP = frozenset(
    """a an the of to in on at for by with from and or not no is are was were be been
    being it its this that these those what which who whom whose when where why how
    does do did done has have had can could should would will shall may might must
    there their them they he she his her we our you your i me my as than then so if
    into about over under per via vs""".split()
)


def _stem(w: str) -> str:
    if len(w) <= 4 or w.isdigit():
        return w
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    """Words (stemmed, stopwords dropped) plus joined compound identifiers.

    'TQ-40' yields tq, 40 and tq40, so it matches 'tq40_datasheet' in a file
    name as well as 'TQ-40' in running text.
    """
    low = text.lower()
    toks = [_stem(w) for w in WORD_RE.findall(low) if w not in STOP]
    for comp in COMPOUND_RE.findall(low):
        joined = re.sub(r"[-_./:#]", "", comp)
        if any(c.isdigit() for c in joined) and len(joined) <= 24:
            toks.append(joined)
    return toks


# --------------------------------------------------------------- identifiers

ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*#?")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def identifiers(text: str) -> set[str]:
    """Distinctive identifier-like tokens: ORR-1847, E7731, TQ-40, B14, 4.3.2.

    Used for multi-hop expansion (an id seen in one file pulls in the file that
    defines it) and for deciding whether a second citation is a real bridge.
    """
    out = set()
    for m in ID_RE.findall(text):
        tok = m.strip("#").strip(".-_/")
        if len(tok) < 3 or len(tok) > 40 or DATE_RE.match(tok):
            continue
        has_d = any(c.isdigit() for c in tok)
        has_a = any(c.isalpha() for c in tok)
        if has_d and has_a:
            letters = sum(c.isalpha() for c in tok)
            if letters == 1 and tok[-1].isalpha() and tok[:-1].replace(".", "").isdigit():
                continue  # "94C", "410W", "4.1T": a value with a unit, not an id
            out.add(tok.upper())
        elif has_d and re.fullmatch(r"\d+(\.\d+){2,}", tok):
            out.add(tok)  # dotted versions: 4.3.2
        elif has_d and re.fullmatch(r"\d{5,}", tok):
            out.add(tok)  # long serial-ish numbers
    return out


# ------------------------------------------------------------------- status

STALE_NAME_RE = re.compile(
    r"withdrawn|superseded|obsolete|deprecated|outdated|retired|do[-_ ]?not[-_ ]?use",
    re.I,
)
STALE_TEXT_RE = re.compile(
    r"\b(withdrawn|superseded\s+by|obsolete|deprecated|no\s+longer\s+(?:valid|current|applies|in\s+effect)|"
    r"do\s+not\s+(?:use|design\s+to|rely)|replaced\s+by|retracted)\b",
    re.I,
)
# "... revision 1 is withdrawn" talks about ANOTHER document, not this one.
OTHER_DOC_RE = re.compile(
    r"(?:revision|rev\.?|version|ver\.?|r|v|issue|edition)\s*\d+[\w.]*\s+(?:is|was|has\s+been|are|were)\s+(?:now\s+)?$",
    re.I,
)
STALE_HEAD_CHARS = 400
VERSION_RE = re.compile(r"(?:^|[-_ .])(r|rev|v|ver|version|draft|issue)[-_ ]?(\d+(?:\.\d+)*)(?=$|[-_ .])", re.I)


def _family_key(path: str) -> tuple[str, str] | None:
    p = PurePosixPath(path)
    stem = p.stem
    m = VERSION_RE.search(stem)
    if not m:
        return None
    base = (stem[: m.start()] + "#" + stem[m.end():]).lower()
    base = STALE_NAME_RE.sub("", base)
    base = re.sub(r"[-_ .]+", "_", base).strip("_")
    return (str(p.parent) + "/" + base + p.suffix.lower(), m.group(2))


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _self_declared_stale(head: str) -> bool:
    for m in STALE_TEXT_RE.finditer(head):
        if not OTHER_DOC_RE.search(head[max(0, m.start() - 40): m.start()]):
            return True
    return False


def doc_status(docs: dict[str, str]) -> dict[str, str]:
    """Return {path: note} for documents that are stale or superseded.

    A document is stale if its name or its opening says so (WITHDRAWN,
    'superseded by', DEPRECATED, ...), or if a sibling with the same name and
    a higher revision number exists (datasheet_r1 vs datasheet_r2).
    """
    notes: dict[str, str] = {}
    for path, text in docs.items():
        name = PurePosixPath(path).name
        dirs = str(PurePosixPath(path).parent)
        if STALE_NAME_RE.search(name) or STALE_NAME_RE.search(dirs):
            notes[path] = "marked withdrawn/superseded in its file name"
        elif _self_declared_stale(text[:STALE_HEAD_CHARS]):
            notes[path] = "states at the top that it is withdrawn/superseded"
    families: dict[str, list[tuple[tuple, str]]] = defaultdict(list)
    for path in docs:
        fk = _family_key(path)
        if fk:
            families[fk[0]].append((_vkey(fk[1]), path))
    for members in families.values():
        if len(members) < 2:
            continue
        members.sort()
        newest = members[-1][1]
        for _, path in members[:-1]:
            notes.setdefault(path, f"older revision; superseded by {newest}")
    return notes


# ------------------------------------------------------------------ chunking

CHUNK_CHARS = 1400
CHUNK_OVERLAP_LINES = 1


def chunk_text(text: str) -> list[str]:
    """Line-aware chunks so a table row or log line is never cut in half."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    # hard-wrap pathological long lines (minified files, one-line dumps)
    flat: list[str] = []
    for ln in lines:
        while len(ln) > CHUNK_CHARS:
            cut = ln.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            flat.append(ln[:cut])
            ln = ln[cut:].lstrip()
        flat.append(ln)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for ln in flat:
        if size + len(ln) + 1 > CHUNK_CHARS and cur:
            chunks.append("\n".join(cur).strip())
            cur = cur[-CHUNK_OVERLAP_LINES:] if CHUNK_OVERLAP_LINES else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        size += len(ln) + 1
    if cur and "\n".join(cur).strip():
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------- BM25


class BM25:
    def __init__(self, token_lists: list[list[str]], k1: float = 1.4, b: float = 0.6):
        self.k1, self.b = k1, b
        self.tfs = [Counter(t) for t in token_lists]
        self.lens = [len(t) for t in token_lists]
        self.n = len(token_lists)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 1.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {
            w: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for w, d in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        q = Counter(query_tokens)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            for w, qn in q.items():
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm) * (1 + 0.25 * (qn - 1))
            out.append(s)
        return out
