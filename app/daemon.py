"""Resident model daemon for MC3.

Started once by the --index pass and left running inside the container, so
each of the ten --query processes pays a socket round-trip instead of a model
load. It knows nothing about retrieval or citations: it turns chat messages
(+ optional images) into text, or scores a fixed set of one-token choices.

Plain HuggingFace transformers on the base image's own ROCm torch -- no vLLM,
no custom kernels, nothing that can swap torch for a CUDA build.

Protocol: unix socket, one JSON object per line each way.
  {"op": "ping"}                                   -> {"ok": true, "device": ...}
  {"op": "generate", "messages": [...], "max_new_tokens": int, "max_time": float}
                                                   -> {"text": str, "secs": float}
  {"op": "choose", "messages": [...], "choices": ["YES", "NO"]}
                                                   -> {"probs": {"YES": p, "NO": q}}
  any failure                                      -> {"error": str}

A message's "content" is a string or a list of parts:
  {"type": "text", "text": str} | {"type": "image", "path": str, "max_pixels": int}
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

SOCK_PATH = os.environ.get("MC3_MODEL_SOCK", "/app/index/model.sock")
MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/qwen3.5-9b")
DEFAULT_MAX_PIXELS = 1_000_000
MIN_SIDE = 64
MAX_IMAGES = 6

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ROCm: never let MIOpen run an exhaustive kernel search on a new shape --
# that is a multi-second stall in the middle of a 30 s question budget.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def log(msg: str) -> None:
    print(f"[daemon {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------- model


def load_processor(model_dir: str):
    """Build the chat/vision processor WITHOUT torchvision.

    The base image ships torch + PIL only, and a PyPI torchvision would drag
    in a CUDA torch. transformers' AutoProcessor for Qwen3-VL/Qwen3.5 insists
    on a torchvision-backed video processor, so we assemble the official
    Qwen3VLProcessor around the PIL image backend and no video processor.
    Verified tensor-for-tensor identical to the torchvision path on images.
    """
    try:
        import torchvision  # noqa: F401
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_dir)
    except Exception as exc:
        log(f"AutoProcessor unavailable ({type(exc).__name__}); using the PIL image backend")
    from transformers import AutoTokenizer
    from transformers.processing_utils import ProcessorMixin
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    orig = ProcessorMixin.check_argument_for_proper_class

    def _allow_none(self, name, arg):
        return None if arg is None else orig(self, name, arg)

    ProcessorMixin.check_argument_for_proper_class = _allow_none
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
        ip = Qwen2VLImageProcessorPil.from_pretrained(model_dir)
        chat_template = None
        for name in ("chat_template.jinja", "chat_template.json"):
            f = Path(model_dir) / name
            if f.exists():
                raw = f.read_text(encoding="utf-8")
                chat_template = json.loads(raw)["chat_template"] if name.endswith(".json") else raw
                break
        return Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=None,
                                chat_template=chat_template or getattr(tok, "chat_template", None))
    finally:
        ProcessorMixin.check_argument_for_proper_class = orig


class Model:
    def __init__(self, model_dir: str):
        import torch

        self.torch = torch
        t0 = time.time()
        if torch.cuda.is_available():  # ROCm presents as cuda
            self.device = torch.device("cuda:0")
            dtype = torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # local development on a Mac
            dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32  # bf16 matmuls on CPU are pathologically slow
            log("WARNING: no GPU visible -- running on CPU (this scores zero in grading)")
        log(f"device={self.device} torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")

        self.processor = self._load_processor(model_dir)
        from transformers import AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.model.to(self.device)  # explicit placement: the VRAM floor is checked
        self.model.eval()
        torch.set_grad_enabled(False)
        tok = self.processor.tokenizer
        self.tokenizer = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated(self.device) / 2**30
            log(f"model on GPU: {used:.1f} GiB allocated")
        log(f"model loaded in {time.time() - t0:.1f}s")

    @staticmethod
    def _load_processor(model_dir: str):
        return load_processor(model_dir)

    # --------------------------------------------------------- inputs

    @staticmethod
    def load_image(path: str, max_pixels: int):
        from PIL import Image, ImageOps

        im = Image.open(path)
        try:
            im.seek(0)  # first frame of animated gif / multi-page tiff
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P", "PA"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        w, h = im.size
        px = w * h
        scale = 1.0
        if px > max_pixels:
            scale = (max_pixels / px) ** 0.5
        elif px < max_pixels / 4:
            # small scans: upscale so the vision tower sees more patches per glyph
            scale = min(2.0, (max_pixels / 4 / px) ** 0.5)
        if min(w, h) * scale < MIN_SIDE:
            scale = MIN_SIDE / min(w, h)
        if abs(scale - 1.0) > 0.02:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        return im

    def build_inputs(self, messages: list[dict], prefix: str = ""):
        images = []
        chat = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                chat.append({"role": msg["role"], "content": content})
                continue
            parts = []
            for part in content or []:
                if part.get("type") == "image":
                    if len(images) >= MAX_IMAGES:
                        continue
                    try:
                        images.append(self.load_image(part["path"], int(part.get("max_pixels") or DEFAULT_MAX_PIXELS)))
                    except Exception as exc:
                        log(f"image load failed {part.get('path')}: {exc}")
                        continue
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": part.get("text", "")})
            chat.append({"role": msg["role"], "content": parts})
        try:
            text = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        text += prefix  # assistant-turn prefill, e.g. '{"evidence": "' to force compact JSON
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        return inputs.to(self.device)

    # ------------------------------------------------------- inference

    def generate(self, messages, max_new_tokens=256, max_time=20.0, prefix="") -> dict:
        t0 = time.time()
        inputs = self.build_inputs(messages, prefix)
        n_in = int(inputs["input_ids"].shape[1])
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_time=float(max_time),
            pad_token_id=self.pad_id,
        )
        new = out[0, n_in:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        text = re.sub(r"(?s)<think>.*?</think>", "", text).replace("<think>", "").strip()
        text = prefix + text
        return {"text": text, "secs": round(time.time() - t0, 2), "in_tokens": n_in, "out_tokens": int(new.shape[0])}

    def choose(self, messages, choices: list[str]) -> dict:
        torch = self.torch
        inputs = self.build_inputs(messages)
        out = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.pad_id,
        )
        logits = out.scores[0][0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = {}
        for c in choices:
            ids = set()
            for variant in (c, " " + c, c.lower(), c.capitalize()):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if toks:
                    ids.add(toks[0])
            probs[c] = float(sum(torch.exp(logprobs[i]).item() for i in ids))
        total = sum(probs.values()) or 1.0
        return {"probs": {k: v / total for k, v in probs.items()}, "mass": total}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def warmup(self, image_path: str | None) -> None:
        """Compile/tune kernels for the shapes queries will use, and measure
        this GPU's prefill and decode speed so the client can size its
        context to the hardware it actually landed on."""
        t0 = time.time()
        self.perf = {"prefill_tps": 0.0, "decode_tps": 0.0}
        line = "Warm-up context line {i}: sensor 12{i} reads 4{i} C, ticket ORR-{i:04d} closed in 4.3.{i}."
        try:
            short = " ".join(line.format(i=i) for i in range(40))
            self.generate([{"role": "user", "content": short + "\nReply OK."}], max_new_tokens=4)
            if image_path:
                self.generate([{"role": "user", "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "Read the text."},
                ]}], max_new_tokens=4)
            if self.device.type == "cpu":
                log("CPU: skipping the speed probe")
                return
            # prefill speed on a realistic ~3k-token prompt (second run = warm)
            long = " ".join(line.format(i=i) for i in range(110))
            msgs = [{"role": "user", "content": long + "\nReply OK."}]
            self.generate(msgs, max_new_tokens=1)
            self._sync()
            t1 = time.time()
            r = self.generate(msgs, max_new_tokens=1)
            self._sync()
            self.perf["prefill_tps"] = round(r["in_tokens"] / max(time.time() - t1, 1e-3), 1)
            # decode speed: 48 forced tokens on a short prompt
            msgs = [{"role": "user", "content": "Count from 1 to 200 separated by spaces."}]
            t1 = time.time()
            r = self.model.generate(**self.build_inputs(msgs), max_new_tokens=48, min_new_tokens=48,
                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                    pad_token_id=self.pad_id)
            self._sync()
            self.perf["decode_tps"] = round(48 / max(time.time() - t1, 1e-3), 1)
        except Exception as exc:
            log(f"warmup failed (continuing): {exc}")
        log(f"warmup done in {time.time() - t0:.1f}s perf={self.perf}")


# ------------------------------------------------------------------ server


def _make_warmup_image() -> str | None:
    try:
        from PIL import Image, ImageDraw

        p = Path(SOCK_PATH).with_name("warmup.png")
        im = Image.new("RGB", (640, 360), "white")
        ImageDraw.Draw(im).text((20, 20), "WARMUP LABEL REV-X1 PIN B00", fill="black")
        im.save(p)
        return str(p)
    except Exception:
        return None


def main() -> int:
    log(f"loading model from {MODEL_DIR}")
    model = Model(MODEL_DIR)
    model.warmup(_make_warmup_image())
    lock = threading.Lock()

    sock_dir = Path(SOCK_PATH).parent
    sock_dir.mkdir(parents=True, exist_ok=True)
    tmp_sock = str(sock_dir / f".model.{os.getpid()}.sock")
    if os.path.exists(tmp_sock):
        os.unlink(tmp_sock)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(tmp_sock)
    os.chmod(tmp_sock, 0o666)
    server.listen(16)
    os.replace(tmp_sock, SOCK_PATH)  # appears atomically, only once we can serve
    log(f"listening on {SOCK_PATH}")

    def handle(conn: socket.socket) -> None:
        with conn:
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    data = conn.recv(1 << 20)
                    if not data:
                        return
                    buf += data
                req = json.loads(buf.decode("utf-8"))
                op = req.get("op", "generate")
                if op == "ping":
                    resp = {"ok": True, "device": str(model.device), "pid": os.getpid(), **model.perf}
                else:
                    with lock:
                        if op == "choose":
                            resp = model.choose(req["messages"], req.get("choices", ["YES", "NO"]))
                        else:
                            resp = model.generate(
                                req["messages"],
                                max_new_tokens=req.get("max_new_tokens", 256),
                                max_time=req.get("max_time", 20.0),
                                prefix=req.get("prefix", ""),
                            )
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except Exception as exc:  # never die: one bad question != nine lost
                log("request failed:\n" + traceback.format_exc())
                try:
                    if model.device.type == "cuda":
                        model.torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    conn.sendall((json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n").encode("utf-8"))
                except OSError:
                    pass

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
