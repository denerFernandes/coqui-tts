# docker/server.py
import os
import uvicorn
import tempfile
import torch
import logging
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variável global para o modelo
tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_model
    logger.info("🚀 Iniciando servidor Coqui TTS...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📱 Dispositivo: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        logger.info("📥 Carregando modelo XTTS-v2...")
        start_time = time.time()
        
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        
        load_time = time.time() - start_time
        logger.info(f"✅ Modelo carregado em {load_time:.2f}s!")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔥 Desligando servidor...")

# Criar app FastAPI
app = FastAPI(
    title="Coqui TTS Server",
    description="Servidor de síntese de voz com XTTS-v2",
    version="1.0.0",
    lifespan=lifespan
)

# CORS para desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Coqui TTS Server", "status": "running"}

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
    
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_info": gpu_info
    }

@app.post("/generate")
async def generate_audio(
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência (.wav)"),
    language: str = Form("pt", description="Código do idioma"),
    temperature: float = Form(0.75, description="Temperatura (0.1-1.0)"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(1.0, description="Penalidade de repetição"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.85, description="Top-p sampling")
):
    if not tts_model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    if not reference_audio.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Formato de áudio não suportado")
    
    if len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Texto muito longo (máximo 1000 caracteres)")
    
    try:
        # Log da requisição
        logger.info(f"🎵 Gerando áudio - Texto: {len(text)} chars, Temp: {temperature}")
        start_time = time.time()
        
        # Salvar áudio de referência temporariamente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        # Gerar áudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name
        
        # Síntese de voz
        tts_model.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p
        )
        
        # Limpar arquivo de referência
        os.unlink(ref_path)
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Áudio gerado em {generation_time:.2f}s")
        
        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename=f"generated_{int(time.time())}.wav",
            headers={"X-Generation-Time": str(generation_time)}
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar áudio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na síntese: {str(e)}")

@app.post("/batch")
async def generate_batch(
    texts: list[str] = Form(..., description="Lista de textos"),
    reference_audio: UploadFile = File(..., description="Áudio de referência"),
    language: str = Form("pt"),
    temperature: float = Form(0.75)
):
    """Gerar múltiplos áudios em lote"""
    if not tts_model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 textos por lote")
    
    results = []
    logger.info(f"🔄 Processando lote de {len(texts)} textos...")
    
    # Salvar áudio de referência
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
        content = await reference_audio.read()
        ref_file.write(content)
        ref_path = ref_file.name
    
    try:
        for i, text in enumerate(texts):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
                    out_path = out_file.name
                
                tts_model.tts_to_file(
                    text=text,
                    speaker_wav=ref_path,
                    language=language,
                    file_path=out_path,
                    temperature=temperature
                )
                
                # Ler arquivo gerado
                with open(out_path, 'rb') as f:
                    audio_data = f.read()
                
                os.unlink(out_path)
                
                results.append({
                    "index": i,
                    "text": text,
                    "success": True,
                    "audio_size": len(audio_data)
                })
                
                logger.info(f"✅ Áudio {i+1}/{len(texts)} gerado")
                
            except Exception as e:
                logger.error(f"❌ Erro no áudio {i+1}: {e}")
                results.append({
                    "index": i,
                    "text": text,
                    "success": False,
                    "error": str(e)
                })
    
    finally:
        os.unlink(ref_path)
    
    successful = len([r for r in results if r["success"]])
    logger.info(f"🎯 Lote concluído: {successful}/{len(texts)} sucessos")
    
    return {"results": results, "summary": {"total": len(texts), "successful": successful}}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )