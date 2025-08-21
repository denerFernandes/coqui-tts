# docker/server.py
import os
import uvicorn
import tempfile
import torch
import logging
import time
import gc
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from contextlib import asynccontextmanager
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variável global para o modelo
tts_model = None

# ========== FUNÇÕES ESSENCIAIS PARA ÁUDIO LIMPO ==========

def preprocess_reference_audio(input_path: str, output_path: str) -> bool:
    """Pré-processa áudio de referência para 16kHz mono (ideal para XTTS v2)"""
    try:
        audio, orig_sr = librosa.load(input_path, sr=None, mono=False)
        
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # Normalização suave
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        logger.info(f"✅ Áudio de referência: {orig_sr}Hz → 16kHz mono")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pré-processamento: {e}")
        return False

def postprocess_to_clean_audio(input_path: str, output_path: str) -> bool:
    """
    Pós-processa para áudio limpo e claro:
    - Converte para 44.1kHz estéreo
    - Normalização final
    """
    try:
        audio, sr = sf.read(input_path)
        logger.info(f"📊 Áudio original: {sr}Hz")
        
        # Resample para 44.1kHz (qualidade padrão universal)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        
        # Converter para estéreo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Normalização final para áudio limpo
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        sf.write(output_path, audio, 44100, subtype='PCM_16')
        logger.info(f"✅ Áudio limpo: 44.1kHz estéreo")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pós-processamento: {e}")
        return False

def cleanup_temp_files(file_paths: list):
    """Remove arquivos temporários"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover {path}: {e}")

# ========== APP SETUP ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_model
    logger.info("🚀 Iniciando servidor Coqui TTS...")
    
    logger.info(f"🔍 CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.warning("⚠️ Usando CPU")
    
    try:
        logger.info("📥 Carregando modelo XTTS-v2...")
        start_time = time.time()
        
        tts_model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2", 
            gpu=torch.cuda.is_available()
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Modelo carregado em {load_time:.2f}s!")
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔥 Desligando servidor...")
    if tts_model:
        del tts_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Criar app FastAPI
app = FastAPI(
    title="Coqui TTS Server",
    description="Servidor de síntese de voz com XTTS-v2 simplificado",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Coqui TTS Server Simplificado", 
        "status": "running",
        "version": "3.0.0",
        "model": "XTTS-v2"
    }

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

# ========== ENDPOINT PRINCIPAL SIMPLIFICADO ==========

@app.post("/generate")
async def generate_audio(
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência"),
    language: str = Form("pt", description="Código do idioma"),
    temperature: float = Form(0.75, description="Temperatura"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(5.0, description="Penalidade de repetição"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.85, description="Top-p sampling"),
    speed: float = Form(1.0, description="Velocidade da fala"),
    enable_text_splitting: bool = Form(True, description="Divisão automática de texto"),
):
    if not tts_model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio")
    
    start_time = time.time()
    ref_path = None
    processed_ref_path = None
    output_path = None
    final_path = None
    
    try:
        logger.info(f"🎵 TTS Simplificado - Texto: {len(text)} chars, Idioma: {language}")
        
        # 1. Salvar áudio de referência
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        # 2. Pré-processar para 16kHz mono (essencial para XTTS v2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_ref_file:
            processed_ref_path = processed_ref_file.name
        
        if not preprocess_reference_audio(ref_path, processed_ref_path):
            raise HTTPException(status_code=400, detail="Falha no pré-processamento")
        
        # 3. Gerar áudio usando streaming do XTTS v2
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name
        
        logger.info("🎤 Gerando áudio com streaming...")

        # Remover pontos duplos, triplos, etc
        text = re.sub(r'\.{2,}', '.', text)

        # Substituir pontos seguidos de espaço por pausa
        text = re.sub(r'\.\s+', '. ', text)

        # Substituir pontos por símbolos alternativos que o TTS entende melhor
        text = text.replace('.', ' ')  # Remove pontos totalmente
        text = text.replace('.', ',')  # Transforma pontos em vírgulas
        text = text.replace('...', ' pausa longa ')  # Reticências explícitas
        
        # Usar split_sentences=True para textos grandes (streaming interno do XTTS)
        tts_model.tts_to_file(
            text=text.strip(),
            speaker_wav=processed_ref_path,
            language=language,
            file_path=output_path,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
            split_sentences=enable_text_splitting  # Deixa o XTTS dividir automaticamente
        )
        
        # 4. Pós-processar para áudio limpo (44.1kHz estéreo)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as final_file:
            final_path = final_file.name
        
        if not postprocess_to_clean_audio(output_path, final_path):
            # Fallback para áudio original se pós-processamento falhar
            final_path = output_path
            logger.warning("⚠️ Usando áudio sem pós-processamento")
        
        # Verificar arquivo final
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise HTTPException(status_code=500, detail="Falha na geração do áudio")
        
        # Estatísticas
        generation_time = time.time() - start_time
        file_size = os.path.getsize(final_path)
        
        try:
            final_audio, final_sr = sf.read(final_path)
            duration = len(final_audio) / final_sr
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - {duration:.1f}s @ {final_sr}Hz")
        except:
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - {file_size} bytes")
        
        return FileResponse(
            final_path,
            media_type="audio/wav",
            filename=f"tts_clean_{int(time.time())}.wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-File-Size": str(file_size),
                "X-Audio-Duration": str(round(duration, 2)) if 'duration' in locals() else "unknown",
                "X-Sample-Rate": str(final_sr) if 'final_sr' in locals() else "44100",
                "X-Clean-Audio": "true",
                "X-Version": "simplified"
            },
            background=lambda: cleanup_temp_files([final_path])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na síntese: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na síntese: {str(e)}")
    
    finally:
        # Limpar arquivos temporários
        cleanup_temp_files([ref_path, processed_ref_path, output_path])

@app.get("/audio-tips")
async def get_audio_tips():
    """Dicas simplificadas para áudio limpo"""
    return {
        "clean_audio_process": [
            "Pré-processamento: Áudio de referência → 16kHz mono",
            "Geração: XTTS v2 com streaming automático para textos grandes", 
            "Pós-processamento: 24kHz → 44.1kHz estéreo (SOM LIMPO)",
            "Resultado: Áudio claro e compatível universalmente"
        ],
        "reference_audio_tips": [
            "5-15 segundos de duração",
            "Voz clara sem ruído de fundo",
            "Qualquer formato (será convertido automaticamente)"
        ],
        "text_tips": [
            "Use pontuação adequada",
            "Para textos grandes: enable_text_splitting=True (padrão)",
            "Sem limite rígido de caracteres (streaming resolve automaticamente)"
        ],
        "parameters": {
            "temperature": "0.1-2.0 (padrão 0.75)",
            "repetition_penalty": "1.0-20.0 (padrão 5.0)", 
            "speed": "0.1-2.0 (padrão 1.0)",
            "top_k": "1-100 (padrão 50)",
            "top_p": "0.1-1.0 (padrão 0.85)",
            "enable_text_splitting": "true/false - streaming para textos grandes"
        },
        "why_clean": "Conversão para 44.1kHz estéreo elimina som abafado do XTTS v2"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )