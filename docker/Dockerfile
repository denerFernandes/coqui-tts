# docker/Dockerfile.simple
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# Configure timezone first
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Install everything in one optimized layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        TTS==0.20.6 \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        python-multipart==0.0.6 \
        soundfile \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache \
    && pip cache purge

COPY server.py .

EXPOSE 8000
CMD ["python", "server.py"]