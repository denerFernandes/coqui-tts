FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN useradd -m -u 1000 coqui
USER coqui
WORKDIR /home/coqui

RUN python -m venv coqui_env
ENV PATH="/home/coqui/coqui_env/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY --chown=coqui:coqui requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=coqui:coqui server.py .

RUN echo "y" | python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" || true

EXPOSE 8000
CMD ["python", "server.py"]