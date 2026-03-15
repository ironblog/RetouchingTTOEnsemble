FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git wget curl ca-certificates \
    ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace/Retouching

RUN python -m pip install --break-system-packages --upgrade pip
RUN python -m pip install --break-system-packages \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --break-system-packages -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/Retouching

CMD ["bash"]
