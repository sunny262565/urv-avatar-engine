FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/MuseTalk \
    URV_AVATAR_ROOT=/runpod-volume/musetalk-results/v15/avatars \
    MUSETALK_HOME=/opt/MuseTalk \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_XET_MAX_CONCURRENT_DOWNLOADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev ffmpeg git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# MuseTalk's documented environment uses PyTorch 2.0.1 with CUDA 11.8.
RUN python3 -m pip install --no-cache-dir \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118

ARG MUSETALK_REF=main
RUN git clone --depth 1 --branch ${MUSETALK_REF} \
      https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk \
    && python3 -m pip install --no-cache-dir -r /opt/MuseTalk/requirements.txt \
    && python3 -m pip install --no-cache-dir hf-xet \
    && python3 -m pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv==2.0.1" \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0"

# Download each Hugging Face artifact once. Pin the face-parser repository to the
# verified revision containing both weights, avoiding the unreliable Drive URL.
RUN python3 - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path("/opt/MuseTalk/models")
items = [
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", root, None),
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", root, None),
    ("stabilityai/sd-vae-ft-mse", "config.json", root / "sd-vae", None),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", root / "sd-vae", None),
    ("openai/whisper-tiny", "config.json", root / "whisper", None),
    ("openai/whisper-tiny", "pytorch_model.bin", root / "whisper", None),
    ("openai/whisper-tiny", "preprocessor_config.json", root / "whisper", None),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", root / "dwpose", None),
    ("ByteDance/LatentSync", "latentsync_syncnet.pt", root / "syncnet", None),
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", root / "face-parse-bisent", "0073b233a5a3c4b1377d4dbf49245017938a72b5"),
    ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth", root / "face-parse-bisent", "0073b233a5a3c4b1377d4dbf49245017938a72b5"),
]
for repository, filename, destination, revision in items:
    destination.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repository,
        filename=filename,
        local_dir=str(destination),
        revision=revision,
    )
PY
RUN python3 - <<'PY'
from pathlib import Path

root = Path("/opt/MuseTalk/models")
required = [
    "musetalkV15/unet.pth", "musetalkV15/musetalk.json",
    "sd-vae/config.json", "sd-vae/diffusion_pytorch_model.bin",
    "whisper/config.json", "whisper/pytorch_model.bin", "whisper/preprocessor_config.json",
    "dwpose/dw-ll_ucoco_384.pth", "syncnet/latentsync_syncnet.pt",
    "face-parse-bisent/79999_iter.pth", "face-parse-bisent/resnet18-5c106cde.pth",
]
missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise RuntimeError(f"MuseTalk model bundle is incomplete: {missing}")
print("MuseTalk model bundle verified")
PY

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN cp /app/urv_stream_adapter.py /opt/MuseTalk/urv_stream_adapter.py
RUN cd /opt/MuseTalk && python3 -c "import musetalk; from urv_stream_adapter import create_renderer"

EXPOSE 8000
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python3","-m","uvicorn","handler:app","--host","0.0.0.0","--port","8000","--workers","1"]
