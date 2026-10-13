# MC3 submission — resident-VLM RAG

**Model:** Qwen3.5-9B (natively multimodal, Feb 2026), bf16, 18.0 GiB → image ≈ 47.5 GiB of 60.
**Runtime:** plain HuggingFace transformers on the base image's own ROCm torch. No vLLM, no custom
kernels, no torchvision. Nothing we install can replace the ROCm torch (the build pins it and
asserts `torch.version.hip` afterwards).

## Build, check, submit (on the AMD dev box)

The weights are not in git. Fetch them first (18 GiB):

```bash
pip install huggingface_hub && hf download Qwen/Qwen3.5-9B --local-dir solution/models/qwen3.5-9b
```

```bash
cd solution
docker build --platform linux/amd64 -t mc3-sub:v1 .
python3 ../selfcheck.py mc3-sub:v1 ../mc3-corpus
python3 ../dev/run_container.py mc3-sub:v1 --gpu        # full ten-question run, scored
python3 ../dev/run_container.py mc3-sub:v1 --gpu --corpus ../dev-run/hard-corpus --questions ../dev-run/hard-questions.json
```

`run_container.py` does what the grader does: `--network none`, `--cap-drop DAC_OVERRIDE`, corpus
copied in with the unreadable file made mode 000 *inside* the container, one `--index`, then one
`docker exec` per question with a hard 30 s timeout, scored as exact citation sets. While it runs,
watch VRAM with `watch -n1 'amd-smi metric --mem | grep USED_VRAM'` (expect ≈ 19–23 GiB).

## How it works

**`--index`** (startup budget, ≤ 540 s by construction)
1. Spawns the model daemon *first*, so the model loads while the corpus is parsed.
2. Walks the corpus in sorted order; every file is parsed in its own try with a timeout. Unreadable,
   encrypted, corrupt, binary and empty files are recorded in `errors.json` and skipped; nothing
   stops the walk. Encrypted PDFs are never decrypted, not even with an empty password: an answer
   that lives only there must be refused.
3. Parsers: PDF (PyMuPDF; text-sparse or image-heavy pages are rendered for OCR), DOCX/PPTX
   (stdlib XML, body order, tables, headers/footers, embedded images), XLSX (openpyxl cached
   values), CSV/TSV, text/code/logs/HTML. Unknown extensions are sniffed by magic bytes. Every
   table row is rendered self-contained (`col: value | col: value`).
4. Every image (standalone pictures, embedded images, rendered PDF pages) is transcribed by the
   VLM, including label→item pairs for diagrams, so images become searchable text.
5. Flags stale documents: WITHDRAWN/SUPERSEDED/DEPRECATED in the name or opening, or an older
   member of a revision family (`spec_v1` vs `spec_v3`, `_r1` vs `_r2`).

**Daemon** (`daemon.py`): loads the model onto the GPU explicitly, warms up the text and vision
paths, measures this GPU's prefill/decode speed, then serves a unix socket. It survives across
`docker exec` calls (new session, stdio detached).

**`--query`** (budget 26.5 s of 30)
1. Writes a valid refusal JSON immediately (atomic rename), so a crash or kill still leaves a
   well-formed file.
2. Retrieval: BM25 with identifier-aware tokens (`TQ-40` → `tq`, `40`, `tq40`). If the whole corpus
   fits the context budget it goes in whole; otherwise top chunks plus **multi-hop expansion**
   (identifiers found in the best hits that are not in the question pull in the chunks that
   define them). The context budget scales with the measured prefill speed of the GPU.
3. Images of top-ranked image documents are attached so the VLM reads the pixels, not only the
   transcript.
4. Generation is forced into compact JSON (`{"evidence": …, "answer": …, "citations": […]}`) by
   prefilling the assistant turn.
5. Deterministic checks (`_finalize_citations`):
   - stale docs are dropped whenever a current doc carries the answer;
   - the answer must be grounded in a cited doc's text; a model that cited only the bridge gets
     the answer-holding doc added;
   - a second citation survives only as a real **bridge**: it shares an identifier with the
     answer's row that the question itself does not already give (Q9 keeps the log; Q4 doesn't);
   - documents the model quoted as evidence become candidates too.
6. A numeric condition in the question ("at 10,000 unit volume") must appear in the cited
   evidence, or the answer is refused.
7. A YES/NO verifier pass (logit probabilities, one token) can veto an unsupported answer into a
   refusal.
8. Numeric answers lose trailing units (`94 C` → `94`); refusal words become `""`.

## Validation done locally (no AMD GPU here)

| Run | Result |
|---|---|
| Sample set, Qwen3.5-4B stand-in on Apple MPS, real 26.5 s budget | 200/200, slowest 19.3 s |
| Harder synthetic corpus (`dev/make_hard_corpus.py`: retrieval mode, revision families, docx table, scanned PDF, image→text multi-hop, refusal traps, volume tiers), 4B stand-in, budget relaxed to 90 s for the Mac | 220/220 (slowest 34 s on MPS; the retrieval-mode multi-hop is the one that needs a real GPU) |
| Container mechanics (`dev/Dockerfile.mechanics`: py3.14, CPU torch, no torchvision, 0.8B stand-in) with `--network none --cap-drop DAC_OVERRIDE` | all contract mechanics pass; every query < 30 s on CPU |

The PIL-backend processor was verified tensor-for-tensor identical to the torchvision one.

## Knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MC3_MODEL_DIR` | `/models/qwen3.5-9b` | model directory |
| `MC3_QUERY_BUDGET_S` | 26.5 | per-question wall budget we plan against |
| `MC3_MIN_CONTEXT_CHARS` / `MC3_MAX_CONTEXT_CHARS` | 14000 / 56000 | adaptive context bounds |
| `MC3_PREFILL_TARGET_S` | 5.0 | seconds of prefill the context is sized for |
| `MC3_CONTEXT_CHARS` | — | force a fixed context size |
| `MC3_OCR_STOP_S` | 500 | stop starting new OCR after this many seconds of `--index` |
| `MC3_SKIP_DAEMON=1` | — | dev only: no model, every answer is a refusal |

Debug traces: `/app/index/queries.log` (one JSON line per question: raw model output, context,
verifier probability, the reason for any refusal), `/app/index/daemon.log`, `/app/index/errors.json`.
