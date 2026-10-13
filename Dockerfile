# MC3 submission image.
#
# Mandated base (checked by layer identity). It ships ROCm torch + PIL only;
# everything we add is pure-python or abi3 wheels that never touch torch.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MIOPEN_FIND_MODE=FAST \
    MC3_MODEL_DIR=/models/qwen3.5-9b

WORKDIR /app

# Pin torch to the exact build the base image ships, so a dependency can never
# replace the ROCm torch with a CUDA wheel from PyPI (which would silently put
# us on the CPU -- an automatic zero). If anything tries, the build fails here.
COPY app/requirements.txt /app/requirements.txt
RUN python3 -c "import torch; print('torch==' + torch.__version__)" > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt \
 && python3 -c "import torch, sys; assert torch.version.hip, 'torch lost its ROCm build'; print('torch', torch.__version__, 'hip', torch.version.hip)"

# Weights ship in the image: there is no network at evaluation time.
# Only the one model directory is copied (see .dockerignore).
COPY models/qwen3.5-9b /models/qwen3.5-9b

COPY app/ /app/

# Fail the BUILD, not the evaluation, if the code or the processor cannot load
# offline in this exact environment.
RUN python3 -m py_compile /app/app.py /app/daemon.py /app/parsers.py /app/retrieval.py \
 && cd /app && python3 -c "import daemon; from transformers import AutoConfig; \
AutoConfig.from_pretrained('/models/qwen3.5-9b'); \
p = daemon.load_processor('/models/qwen3.5-9b'); \
print('processor OK:', type(p).__name__, type(p.image_processor).__name__)" \
 && python3 -c "import pymupdf, openpyxl, pypdf; print('parsers OK')"

RUN mkdir -p /app/corpus /app/output /app/index && chmod 777 /app/output /app/index

# No service, no port, no entrypoint: the harness execs into a running container.
CMD ["sleep", "infinity"]
