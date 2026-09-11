FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
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

# MuseTalk's upstream downloader has changed over time and some revisions do
# not fetch the DWPose checkpoint.  Re-fetch it explicitly when absent, then
# fail the image build if the checkpoint is still unavailable.
RUN python3 -c "from pathlib import Path; import urllib.request; p=Path('/opt/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth'); p.parent.mkdir(parents=True, exist_ok=True); p.exists() or urllib.request.urlretrieve('https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth', p)" \
    && test -s /opt/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN cp /app/urv_stream_adapter.py /opt/MuseTalk/urv_stream_adapter.py

EXPOSE 8000
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python3","-m","uvicorn","handler:app","--host","0.0.0.0","--port","8000","--workers","1"]
