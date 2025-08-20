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

def validate_reference_audio(audio_path: str) -> dict:
    """Validar e analisar áudio de referência"""
    try:
        audio_data, sample_rate = sf.read(audio_path)
        
        # Converter para mono se necessário
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Validações
        if duration < 3.0:
            return {"valid": False, "error": "Áudio de referência muito curto (mínimo 3s)"}
        
        if duration > 30.0:
            return {"valid": False, "error": "Áudio de referência muito longo (máximo 30s)"}
        
        if rms < 0.01:
            return {"valid": False, "error": "Áudio de referência muito baixo (aumente o volume)"}
        
        return {
            "valid": True, 
            "duration": duration,
            "sample_rate": sample_rate,
            "rms": rms,
            "recommendations": get_audio_recommendations(duration, rms)
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Erro ao analisar áudio: {str(e)}"}

def get_audio_recommendations(duration: float, rms: float) -> list:
    """Gerar recomendações baseadas na análise do áudio"""
    recommendations = []
    
    if duration < 5:
        recommendations.append("Use áudio mais longo (5-15s) para melhor qualidade")
    
    if rms < 0.05:
        recommendations.append("Áudio muito baixo - use áudio com voz mais forte")
    
    if rms > 0.3:
        recommendations.append("Áudio pode estar muito alto - normalize o volume")
    
    return recommendations

def remove_silence_from_audio(audio_path: str, output_path: str) -> bool:
    """Remover silêncios excessivos do áudio gerado"""
    try:
        audio_data, sample_rate = sf.read(audio_path)
        
        # Parâmetros para detecção de silêncio
        silence_threshold = 0.01
        min_silence_duration = 0.5  # segundos
        min_silence_samples = int(min_silence_duration * sample_rate)
        
        # Detectar segmentos não-silenciosos
        audio_abs = np.abs(audio_data)
        non_silent_indices = audio_abs > silence_threshold
        
        # Encontrar início e fim de segmentos de fala
        speech_segments = []
        in_speech = False
        start = 0
        
        for i, is_speech in enumerate(non_silent_indices):
            if is_speech and not in_speech:
                start = max(0, i - int(0.1 * sample_rate))  # Padding de 0.1s
                in_speech = True
            elif not is_speech and in_speech:
                end = min(len(audio_data), i + int(0.1 * sample_rate))  # Padding de 0.1s
                if end - start > min_silence_samples:
                    speech_segments.append((start, end))
                in_speech = False
        
        # Adicionar último segmento se necessário
        if in_speech:
            speech_segments.append((start, len(audio_data)))
        
        # Concatenar segmentos com pausas menores
        if speech_segments:
            processed_audio = []
            for i, (start, end) in enumerate(speech_segments):
                processed_audio.append(audio_data[start:end])
                # Adicionar pausa curta entre segmentos (exceto no último)
                if i < len(speech_segments) - 1:
                    pause_samples = int(0.3 * sample_rate)  # 0.3s de pausa
                    processed_audio.append(np.zeros(pause_samples))
            
            final_audio = np.concatenate(processed_audio)
            sf.write(output_path, final_audio, sample_rate)
            logger.info(f"🎵 Silêncios removidos: {len(audio_data)/sample_rate:.1f}s → {len(final_audio)/sample_rate:.1f}s")
            return True
        else:
            # Se não encontrar fala, manter original
            sf.write(output_path, audio_data, sample_rate)
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Erro ao remover silêncios: {e}")
        return False

def optimize_text_for_speech(text: str, language: str = "pt") -> str:
    """Otimizar texto para melhor síntese de voz"""
    # Adicionar pontuação para pausas naturais
    text = text.strip()
    
    # Adicionar pontos finais se não houver
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Substituir abreviações comuns (português)
    if language == "pt":
        replacements = {
            ' dr ': ' doutor ',
            ' dra ': ' doutora ',
            ' prof ': ' professor ',
            ' sra ': ' senhora ',
            ' sr ': ' senhor ',
            'vc': 'você',
            'pq': 'porque',
            'td': 'tudo',
            'tbm': 'também'
        }
        text_lower = text.lower()
        for abbrev, full in replacements.items():
            text = text.replace(abbrev, full)
            text_lower = text_lower.replace(abbrev, full)
    
    return text

def add_warmup_context(text: str, language: str = "pt") -> tuple:
    """Adicionar contexto de aquecimento para melhorar início do áudio"""
    warmup_texts = {
        "pt": [
            "Olá.",
            "Bem-vindos.",
            "Atenção por favor."
        ],
        "en": [
            "Hello.",
            "Welcome.",
            "Attention please."
        ],
        "es": [
            "Hola.",
            "Bienvenidos.",
            "Atención por favor."
        ]
    }
    
    # Selecionar frase de aquecimento baseada no idioma
    warmup_phrase = warmup_texts.get(language, warmup_texts["en"])[0]
    
    # Criar texto com aquecimento
    warmed_text = f"{warmup_phrase} {text}"
    
    # Calcular onde cortar o aquecimento no áudio final
    warmup_word_count = len(warmup_phrase.split())
    
    return warmed_text, warmup_word_count

def trim_warmup_from_audio(audio_path: str, output_path: str, warmup_word_count: int, sample_rate: int = 22050) -> bool:
    """Remover o aquecimento do início do áudio"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Estimar duração do aquecimento (aproximadamente 0.6s por palavra + 0.3s de pausa)
        estimated_warmup_duration = (warmup_word_count * 0.6) + 0.3
        warmup_samples = int(estimated_warmup_duration * sr)
        
        # Adicionar pequeno fade-in para evitar cliques
        fade_samples = int(0.05 * sr)  # 50ms fade-in
        
        if warmup_samples < len(audio_data):
            # Cortar aquecimento
            trimmed_audio = audio_data[warmup_samples:]
            
            # Aplicar fade-in suave
            if len(trimmed_audio) > fade_samples:
                fade_curve = np.linspace(0, 1, fade_samples)
                trimmed_audio[:fade_samples] *= fade_curve
            
            sf.write(output_path, trimmed_audio, sr)
            logger.info(f"🎬 Aquecimento removido: {warmup_samples/sr:.2f}s cortados do início")
            return True
        else:
            # Se estimativa foi errada, manter original
            sf.write(output_path, audio_data, sr)
            logger.warning("⚠️ Não foi possível remover aquecimento - mantendo áudio original")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao remover aquecimento: {e}")
        return False

def improve_audio_start(audio_path: str, output_path: str) -> bool:
    """Melhorar qualidade do início do áudio"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Parâmetros para melhoria do início
        start_improvement_duration = 1.0  # primeiros 1 segundo
        start_samples = int(start_improvement_duration * sr)
        
        if len(audio_data) > start_samples:
            # Aplicar filtro suave no início para reduzir artefatos
            start_segment = audio_data[:start_samples].copy()
            
            # Suavização com média móvel
            window_size = int(0.01 * sr)  # janela de 10ms
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                start_segment = np.convolve(start_segment, kernel, mode='same')
            
            # Aplicar fade-in muito suave
            fade_samples = int(0.1 * sr)  # 100ms
            if fade_samples < len(start_segment):
                fade_curve = np.linspace(0.3, 1.0, fade_samples)  # começar em 30% do volume
                start_segment[:fade_samples] *= fade_curve
            
            # Combinar início melhorado com resto do áudio
            improved_audio = audio_data.copy()
            improved_audio[:start_samples] = start_segment
            
            sf.write(output_path, improved_audio, sr)
            logger.info("🎛️ Início do áudio melhorado")
            return True
        else:
            sf.write(output_path, audio_data, sr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao melhorar início do áudio: {e}")
        return False

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
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência (.wav)"),
    language: str = Form("pt", description="Código do idioma"),
    temperature: float = Form(0.85, description="Temperatura (0.7-0.95) - maior = mais expressiva"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(5.0, description="Penalidade de repetição (2.0-10.0)"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.85, description="Top-p sampling"),
    speed: float = Form(1.0, description="Velocidade da fala (0.8-1.2)"),
    enable_text_splitting: bool = Form(True, description="Dividir texto em sentenças"),
    remove_silence: bool = Form(True, description="Remover silêncios excessivos"),
    improve_start: bool = Form(True, description="Melhorar qualidade do início do áudio"),
    use_warmup: bool = Form(True, description="Usar aquecimento para melhorar início")
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
    
    # Validar parâmetros melhorados
    if not (0.7 <= temperature <= 0.95):
        raise HTTPException(status_code=400, detail="Temperatura deve estar entre 0.7 e 0.95 para melhor qualidade")
    
    if not (2.0 <= repetition_penalty <= 10.0):
        raise HTTPException(status_code=400, detail="Repetition penalty deve estar entre 2.0 e 10.0")
    
    if not (0.8 <= speed <= 1.2):
        raise HTTPException(status_code=400, detail="Velocidade deve estar entre 0.8 e 1.2")
    
    ref_path = None
    out_path = None
    
    try:
        # Log da requisição
        logger.info(f"🎵 Gerando áudio - Texto: {len(text)} chars, Idioma: {language}, Temp: {temperature}")
        start_time = time.time()
        
        # Otimizar texto para melhor síntese
        optimized_text = optimize_text_for_speech(text, language)
        if optimized_text != text:
            logger.info(f"📝 Texto otimizado para síntese")
        
        # Adicionar aquecimento se habilitado
        warmup_word_count = 0
        if use_warmup:
            warmed_text, warmup_word_count = add_warmup_context(optimized_text, language)
            logger.info(f"🔥 Aquecimento adicionado - {warmup_word_count} palavras")
            final_text = warmed_text
        else:
            final_text = optimized_text
        
        # Salvar áudio de referência temporariamente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        logger.info(f"📁 Áudio de referência salvo: {ref_path}")
        
        # Validar áudio de referência
        validation = validate_reference_audio(ref_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Áudio de referência inválido: {validation['error']}")
        
        logger.info(f"✅ Áudio de referência válido - Duração: {validation['duration']:.1f}s")
        if validation.get("recommendations"):
            for rec in validation["recommendations"]:
                logger.info(f"💡 {rec}")
        
        # Gerar áudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name
        
        # Síntese de voz com parâmetros otimizados
        try:
            logger.info(f"🎤 Iniciando síntese com parâmetros otimizados...")
            
            tts_model.tts_to_file(
                text=final_text,  # Usar texto com aquecimento se habilitado
                speaker_wav=ref_path,
                language=language,
                file_path=out_path,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
                split_sentences=enable_text_splitting
            )
            
        except AttributeError as e:
            if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                logger.error("❌ Erro de compatibilidade detectado! Verifique as versões do coqui-tts e transformers")
                raise HTTPException(
                    status_code=500, 
                    detail="Erro de compatibilidade: Atualize para coqui-tts>=0.27.0 e transformers<=4.46.2"
                )
            raise
        
        # Pós-processamento em etapas
        current_path = out_path
        processing_steps = []
        
        # Etapa 1: Remover aquecimento se foi usado
        if use_warmup and warmup_word_count > 0:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as trimmed_file:
                trimmed_path = trimmed_file.name
            
            try:
                audio_data, sr = sf.read(current_path)
                if trim_warmup_from_audio(current_path, trimmed_path, warmup_word_count, sr):
                    os.unlink(current_path)
                    current_path = trimmed_path
                    processing_steps.append("aquecimento removido")
                else:
                    os.unlink(trimmed_path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover aquecimento: {e}")
                os.unlink(trimmed_path)
        
        # Etapa 2: Melhorar início do áudio
        if improve_start:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as improved_file:
                improved_path = improved_file.name
            
            if improve_audio_start(current_path, improved_path):
                os.unlink(current_path)
                current_path = improved_path
                processing_steps.append("início melhorado")
            else:
                os.unlink(improved_path)
        
        # Etapa 3: Remover silêncios se solicitado
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_file:
                processed_path = processed_file.name
            
            if remove_silence_from_audio(current_path, processed_path):
                os.unlink(current_path)
                current_path = processed_path
                processing_steps.append("silêncios removidos")
            else:
                os.unlink(processed_path)
        
        # Verificar se o arquivo foi gerado
        if not os.path.exists(current_path) or os.path.getsize(current_path) == 0:
            raise HTTPException(status_code=500, detail="Falha na geração do áudio")
        
        generation_time = time.time() - start_time
        file_size = os.path.getsize(current_path)
        
        # Informações do áudio gerado
        try:
            final_audio, final_sr = sf.read(current_path)
            final_duration = len(final_audio) / final_sr
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - Tamanho: {file_size} bytes - Duração: {final_duration:.1f}s")
            if processing_steps:
                logger.info(f"🎛️ Pós-processamento aplicado: {', '.join(processing_steps)}")
        except:
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - Tamanho: {file_size} bytes")
        
        return FileResponse(
            current_path,
            media_type="audio/wav",
            filename=f"generated_{int(time.time())}.wav",
            headers={
                "X-Generation-Time": str(generation_time),
                "X-File-Size": str(file_size),
                "X-Audio-Duration": str(final_duration) if 'final_duration' in locals() else "unknown",
                "X-Processing-Steps": ",".join(processing_steps) if processing_steps else "none",
                "X-Warmup-Used": "true" if use_warmup else "false",
                "X-Start-Improved": "true" if improve_start else "false"
            },
            background=lambda: os.unlink(current_path) if os.path.exists(current_path) else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao gerar áudio: {str(e)}")
        logger.error(f"Tipo de erro: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Erro na síntese: {str(e)}")
    
    finally:
        # Limpar apenas arquivo de referência (não o arquivo de saída que será enviado)
        if ref_path and os.path.exists(ref_path):
            try:
                os.unlink(ref_path)
                logger.debug(f"🗑️ Arquivo de referência removido: {ref_path}")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover arquivo de referência {ref_path}: {e}")
        
        # NOTA: out_path é removido automaticamente pelo FastAPI FileResponse

@app.post("/generate-preset")
async def generate_audio_preset(
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência (.wav)"),
    language: str = Form("pt", description="Código do idioma"),
    preset: str = Form("balanced", description="Preset de qualidade: energetic, balanced, calm, custom")
):
    """Gerar áudio com presets otimizados para diferentes estilos"""
    
    # Definir presets
    presets = {
        "energetic": {
            "temperature": 0.90,
            "repetition_penalty": 7.0,
            "speed": 1.1,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True
        },
        "balanced": {
            "temperature": 0.85,
            "repetition_penalty": 5.0,
            "speed": 1.0,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True
        },
        "calm": {
            "temperature": 0.80,
            "repetition_penalty": 3.0,
            "speed": 0.95,
            "remove_silence": False,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": False
        },
        "expressive": {
            "temperature": 0.95,
            "repetition_penalty": 8.0,
            "speed": 1.05,
            "remove_silence": True,
            "enable_text_splitting": False,
            "improve_start": True,
            "use_warmup": True
        },
        "professional": {
            "temperature": 0.87,
            "repetition_penalty": 6.0,
            "speed": 1.0,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True
        }
    }
    
    if preset not in presets:
        raise HTTPException(status_code=400, detail=f"Preset inválido. Use: {', '.join(presets.keys())}")
    
    config = presets[preset]
    logger.info(f"🎭 Usando preset '{preset}' com configurações: {config}")
    
    # Chamar função principal com configurações do preset
    return await generate_audio(
        text=text,
        reference_audio=reference_audio,
        language=language,
        temperature=config["temperature"],
        length_penalty=1.0,
        repetition_penalty=config["repetition_penalty"],
        top_k=50,
        top_p=0.85,
        speed=config["speed"],
        enable_text_splitting=config["enable_text_splitting"],
        remove_silence=config["remove_silence"],
        improve_start=config["improve_start"],
        use_warmup=config["use_warmup"]
    )

@app.get("/audio-tips")
async def get_audio_tips():
    """Dicas para melhorar a qualidade do áudio"""
    return {
        "reference_audio_tips": [
            "Use áudio de 5-15 segundos de duração",
            "Voz clara e bem articulada",
            "Sem ruído de fundo ou eco",
            "Volume adequado (não muito baixo/alto)",
            "Emoção/energia desejada na voz",
            "Preferencialmente em .wav 22kHz ou 44kHz"
        ],
        "text_tips": [
            "Use pontuação adequada (. ! ?)",
            "Evite texto muito longo (máx 1000 chars)",
            "Escreva números por extenso (2023 → dois mil e vinte e três)",
            "Evite abreviações (Dr → Doutor)",
            "Use frases completas"
        ],
        "parameter_tips": {
            "temperature": "0.85-0.95 para voz mais expressiva",
            "repetition_penalty": "5.0-8.0 para evitar repetições",
            "speed": "1.0-1.1 para voz mais dinâmica",
            "remove_silence": "true para áudio mais limpo",
            "improve_start": "true para melhorar qualidade do início",
            "use_warmup": "true para eliminar problemas no início do áudio"
        },
        "presets": {
            "energetic": "Voz animada e dinâmica com início otimizado",
            "balanced": "Equilíbrio entre naturalidade e expressividade",
            "calm": "Voz mais suave e relaxada",
            "expressive": "Máxima expressividade e emoção",
            "professional": "Ideal para narrações e apresentações corporativas"
        },
        "start_quality_tips": [
            "Use 'use_warmup=true' para melhorar drasticamente o início",
            "Combine com 'improve_start=true' para máxima qualidade",
            "O aquecimento adiciona contexto que melhora a síntese",
            "O processamento remove automaticamente o aquecimento",
            "Resultado: início limpo e de alta qualidade"
        ]
    }
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