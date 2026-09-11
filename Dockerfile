FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/MuseTalk \
    URV_AVATAR_ROOT=/runpod-volume/musetalk-results/v15/avatars \
    MUSETALK_HOME=/opt/MuseTalk

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
    && python3 -m pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv==2.0.1" \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0"

# Keep weights in the immutable image to avoid downloading them after scale-to-zero.
RUN cd /opt/MuseTalk && sh ./download_weights.sh

# MuseTalk's upstream downloader has changed over time and some revisions omit
# individual dependency files. Fetch every runtime bundle idempotently and fail
# the image build if any file needed by real-time preprocessing is unavailable.
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download

root = "/opt/MuseTalk/models"
bundles = [
    ("TMElyralab/MuseTalk", root, ["musetalkV15/unet.pth", "musetalkV15/musetalk.json"]),
    ("stabilityai/sd-vae-ft-mse", f"{root}/sd-vae", ["config.json", "diffusion_pytorch_model.bin"]),
    ("openai/whisper-tiny", f"{root}/whisper", ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ("yzd-v/DWPose", f"{root}/dwpose", ["dw-ll_ucoco_384.pth"]),
    ("ByteDance/LatentSync", f"{root}/syncnet", ["latentsync_syncnet.pt"]),
    ("ManyOtherFunctions/face-parse-bisent", f"{root}/face-parse-bisent", ["79999_iter.pth", "resnet18-5c106cde.pth"]),
]
for repository, destination, patterns in bundles:
    snapshot_download(
        repo_id=repository,
        local_dir=destination,
        allow_patterns=patterns,
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
