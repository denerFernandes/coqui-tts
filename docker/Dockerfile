# docker/Dockerfile.optimized
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Configurar timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
        build-essential \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN python -m pip install --upgrade pip

# Instalar dependências Python específicas (versões fixas para evitar conflitos)
RUN pip install coqui-tts==0.27.0
RUN pip install transformers==4.46.2  
RUN pip install fastapi==0.116.1
RUN pip install uvicorn==0.35.0
RUN pip install python-multipart==0.0.20
RUN pip install soundfile==0.13.1
RUN pip install numpy==1.24.3
RUN pip install scipy==1.10.1
RUN pip install requests==2.31.0

# Pre-download do modelo (opcional - comentar se quiser baixar no startup)
RUN echo "y" | python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" || true

# Copiar aplicação
COPY server.py .

# Criar usuário não-root (opcional, para segurança)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]