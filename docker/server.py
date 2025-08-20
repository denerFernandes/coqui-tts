# docker/server.py
import os
import uvicorn
import tempfile
import torch
import logging
import time
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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

def cleanup_file(file_path: str):
    """Remove arquivo após envio"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"🗑️ Arquivo removido após envio: {file_path}")
    except Exception as e:
        logger.warning(f"⚠️ Falha ao remover arquivo após envio {file_path}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_model
    logger.info("🚀 Iniciando servidor Coqui TTS...")
    
    # Verificação detalhada de GPU
    logger.info(f"🔍 PyTorch version: {torch.__version__}")
    logger.info(f"🔍 CUDA compiled version: {torch.version.cuda}")
    logger.info(f"🔍 CUDA available: {torch.cuda.is_available()}")
    logger.info(f"🔍 CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"📱 Dispositivo: {device}")
        logger.info(f"🎮 GPU: {gpu_name}")
        logger.info(f"💾 VRAM Total: {gpu_memory:.1f}GB")
        
        # Teste simples de GPU
        try:
            test_tensor = torch.randn(10, 10).cuda()
            logger.info(f"✅ Teste de GPU bem-sucedido")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ Erro no teste de GPU: {e}")
            device = "cpu"
    else:
        device = "cpu"
        logger.warning("⚠️ GPU não disponível - usando CPU")
        logger.info("📱 Dispositivo: cpu")
        
        # Dicas para resolver problema de GPU
        logger.info("💡 Para usar GPU:")
        logger.info("   - Certifique-se que nvidia-docker está instalado")
        logger.info("   - Use: docker run --gpus all ...")
        logger.info("   - Ou: nvidia-docker run ...")
    
    try:
        logger.info("📥 Carregando modelo XTTS-v2...")
        start_time = time.time()
        
        # Forçar uso de GPU se disponível
        use_gpu = torch.cuda.is_available()
        logger.info(f"🎯 Carregando modelo com GPU: {use_gpu}")
        
        # Carregar modelo com tratamento de erro melhorado
        tts_model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2", 
            gpu=use_gpu
        )
        
        # Verificar se modelo foi carregado na GPU
        if use_gpu and hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'tts_model'):
            model_device = next(tts_model.synthesizer.tts_model.parameters()).device
            logger.info(f"🎯 Modelo carregado no dispositivo: {model_device}")
        
        load_time = time.time() - start_time
        logger.info(f"✅ Modelo carregado em {load_time:.2f}s!")
        
        # Verificar se o modelo tem os métodos necessários
        if not hasattr(tts_model, 'tts_to_file'):
            raise AttributeError("Modelo não possui método tts_to_file")
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        logger.error(f"Tipo de erro: {type(e).__name__}")
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
    description="Servidor de síntese de voz com XTTS-v2",
    version="2.0.0",
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
    return {
        "message": "Coqui TTS Server", 
        "status": "running",
        "version": "2.0.0",
        "model": "XTTS-v2"
    }

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "gpu_memory_free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f}GB"
        }
    
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_info": gpu_info,
        "temp_dir": str(tempfile.gettempdir())
    }

@app.post("/generate")
async def generate_audio(
    background_tasks: BackgroundTasks,
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
    
    # Validações melhoradas
    if not reference_audio.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo de áudio não fornecido")
    
    allowed_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    if not reference_audio.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de áudio não suportado. Use: {', '.join(allowed_extensions)}"
        )
    
    if len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Texto muito longo (máximo 1000 caracteres)")
    
    # Validar parâmetros
    if not (0.1 <= temperature <= 1.0):
        raise HTTPException(status_code=400, detail="Temperatura deve estar entre 0.1 e 1.0")
    
    ref_path = None
    out_path = None
    
    try:
        # Log da requisição
        logger.info(f"🎵 Gerando áudio - Texto: {len(text)} chars, Idioma: {language}, Temp: {temperature}")
        start_time = time.time()
        
        # Salvar áudio de referência temporariamente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        logger.info(f"📁 Áudio de referência salvo: {ref_path}")
        
        # Gerar áudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name
        
        # Síntese de voz com tratamento de erro específico
        try:
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
        except AttributeError as e:
            if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                logger.error("❌ Erro de compatibilidade detectado! Verifique as versões do coqui-tts e transformers")
                raise HTTPException(
                    status_code=500, 
                    detail="Erro de compatibilidade: Atualize para coqui-tts>=0.27.0 e transformers<=4.46.2"
                )
            raise
        
        # Verificar se o arquivo foi gerado
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise HTTPException(status_code=500, detail="Falha na geração do áudio")
        
        generation_time = time.time() - start_time
        file_size = os.path.getsize(out_path)
        logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - Tamanho: {file_size} bytes")
        
        # Agendar limpeza do arquivo após envio
        background_tasks.add_task(cleanup_file, out_path)
        
        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename=f"generated_{int(time.time())}.wav",
            headers={
                "X-Generation-Time": str(generation_time),
                "X-File-Size": str(file_size)
            },
            background=lambda: os.unlink(out_path) if os.path.exists(out_path) else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao gerar áudio: {str(e)}")
        logger.error(f"Tipo de erro: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Erro na síntese: {str(e)}")
    
    finally:
        # Limpar apenas arquivo de referência - out_path será auto-removido
        if ref_path and os.path.exists(ref_path):
            try:
                os.unlink(ref_path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover arquivo de referência: {e}")
        
        # NOTA: out_path é removido automaticamente pelo FastAPI FileResponse

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
    
    if not texts:
        raise HTTPException(status_code=400, detail="Lista de textos não pode estar vazia")
    
    results = []
    ref_path = None
    logger.info(f"🔄 Processando lote de {len(texts)} textos...")
    
    # Salvar áudio de referência
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        for i, text in enumerate(texts):
            out_path = None
            try:
                if not text.strip():
                    results.append({
                        "index": i,
                        "text": text,
                        "success": False,
                        "error": "Texto vazio"
                    })
                    continue
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
                    out_path = out_file.name
                
                tts_model.tts_to_file(
                    text=text,
                    speaker_wav=ref_path,
                    language=language,
                    file_path=out_path,
                    temperature=temperature
                )
                
                # Verificar se arquivo foi gerado
                if os.path.exists(out_path):
                    file_size = os.path.getsize(out_path)
                    results.append({
                        "index": i,
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "success": True,
                        "audio_size": file_size
                    })
                    logger.info(f"✅ Áudio {i+1}/{len(texts)} gerado - {file_size} bytes")
                else:
                    results.append({
                        "index": i,
                        "text": text,
                        "success": False,
                        "error": "Arquivo não foi gerado"
                    })
                
            except Exception as e:
                logger.error(f"❌ Erro no áudio {i+1}: {e}")
                results.append({
                    "index": i,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "success": False,
                    "error": str(e)
                })
            
            finally:
                if out_path and os.path.exists(out_path):
                    try:
                        os.unlink(out_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Falha ao remover {out_path}: {e}")
    
    finally:
        if ref_path and os.path.exists(ref_path):
            try:
                os.unlink(ref_path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover arquivo de referência: {e}")
    
    successful = len([r for r in results if r["success"]])
    total_size = sum(r.get("audio_size", 0) for r in results if r["success"])
    
    logger.info(f"🎯 Lote concluído: {successful}/{len(texts)} sucessos - Total: {total_size} bytes")
    
    return {
        "results": results, 
        "summary": {
            "total": len(texts), 
            "successful": successful,
            "failed": len(texts) - successful,
            "total_audio_size": total_size
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )