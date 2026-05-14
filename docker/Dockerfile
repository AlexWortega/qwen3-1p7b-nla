FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    WANDB_DIR=/cache/wandb \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
        "transformers>=4.46,<5.0" \
        "peft>=0.13,<0.16" \
        "accelerate>=1.0,<2.0" \
        "datasets>=3.0,<4.0" \
        "safetensors>=0.4" \
        "wandb>=0.18" \
        "openai>=1.50" \
        "anthropic>=0.40" \
        "pyarrow>=15.0" \
        "orjson>=3.10" \
        "python-dotenv>=1.0" \
        "pyyaml>=6.0" \
        "tqdm>=4.66" \
        "httpx>=0.27" \
        "claude-monitor-sdk"

COPY nla/ ./nla/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-c", "import torch; print('cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"]
