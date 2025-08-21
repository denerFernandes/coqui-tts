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
from scipy import signal
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

# ========== FUNÇÕES OTIMIZADAS (RECOMENDAÇÕES CHATGPT) ==========

def preprocess_reference_audio(input_path: str, output_path: str) -> bool:
    """
    Pré-processa o áudio de referência para formato ideal do XTTS v2:
    - 16 kHz mono PCM
    - Normalização de volume
    - Remoção de ruído básica
    """
    try:
        # Carregar áudio original
        audio, orig_sr = librosa.load(input_path, sr=None, mono=False)
        
        # Converter para mono se necessário
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Resample para 16 kHz (ideal para XTTS v2)
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # Normalização suave
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Remoção básica de ruído (filtro passa-alta leve)
        b, a = signal.butter(3, 80, btype='high', fs=16000)
        audio = signal.filtfilt(b, a, audio)
        
        # Salvar como 16-bit PCM
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        
        logger.info(f"✅ Áudio de referência processado: {orig_sr}Hz → 16kHz mono")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pré-processamento: {e}")
        return False

def postprocess_generated_audio(input_path: str, output_path: str) -> bool:
    """
    Pós-processa o áudio gerado:
    - Converte de 24kHz para 44.1kHz
    - Mono para estéreo
    - Normalização final
    """
    try:
        # Carregar áudio gerado (normalmente 24kHz mono do XTTS v2)
        audio, sr = sf.read(input_path)
        
        logger.info(f"📊 Áudio original: {sr}Hz, {audio.shape}")
        
        # Resample para 44.1kHz (qualidade padrão)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        
        # Converter para estéreo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Normalização final suave
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        # Salvar como 44.1kHz estéreo 16-bit
        sf.write(output_path, audio, 44100, subtype='PCM_16')
        
        logger.info(f"✅ Áudio pós-processado: 44.1kHz estéreo")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pós-processamento: {e}")
        return False

def cleanup_temp_files(file_paths: list):
    """Remove arquivos temporários de forma segura"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.debug(f"🗑️ Arquivo removido: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover {path}: {e}")

# ========== FUNÇÕES ORIGINAIS (MANTIDAS PARA COMPATIBILIDADE) ==========

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

def get_character_limits() -> dict:
    """Limites de caracteres por idioma para XTTS"""
    return {
        "pt": 160,  # Português - limite conservador para melhor qualidade
        "en": 250,  # Inglês
        "es": 220,  # Espanhol
        "fr": 230,  # Francês
        "de": 180,  # Alemão
        "it": 210,  # Italiano
        "pl": 190,  # Polonês
        "tr": 200,  # Turco
        "ru": 180,  # Russo
        "nl": 200,  # Holandês
        "cs": 190,  # Tcheco
        "ar": 150,  # Árabe
        "zh": 100,  # Chinês
        "ja": 120,  # Japonês
        "hi": 150,  # Hindi
        "ko": 130   # Coreano
    }

def smart_text_split(text: str, language: str = "pt", max_chars: int = None) -> list:
    """Dividir texto inteligentemente respeitando limites por idioma"""
    limits = get_character_limits()
    if max_chars is None:
        max_chars = limits.get(language, 160)
    
    # Se texto está dentro do limite, retornar como está
    if len(text) <= max_chars:
        return [text]
    
    # Dividir por sentenças primeiro
    import re
    
    # Padrões de divisão por idioma
    sentence_patterns = {
        "pt": r'[.!?]+\s+',
        "en": r'[.!?]+\s+',
        "es": r'[.!?]+\s+',
        "fr": r'[.!?]+\s+',
        "de": r'[.!?]+\s+',
        "it": r'[.!?]+\s+',
    }
    
    pattern = sentence_patterns.get(language, r'[.!?]+\s+')
    sentences = re.split(pattern, text.strip())
    
    # Reagrupar sentenças em chunks menores que o limite
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Adicionar pontuação se não houver
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Se sentença sozinha já excede limite, dividir por vírgulas
        if len(sentence) > max_chars:
            sub_parts = sentence.split(',')
            for part in sub_parts:
                part = part.strip()
                if len(part) > max_chars:
                    # Se ainda muito grande, dividir forçadamente
                    for i in range(0, len(part), max_chars - 10):
                        chunk_part = part[i:i + max_chars - 10].strip()
                        if chunk_part:
                            chunks.append(chunk_part + '.')
                else:
                    if len(current_chunk + part) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + ', '
                    else:
                        current_chunk += part + ', '
        else:
            # Verificar se pode adicionar à chunk atual
            test_chunk = current_chunk + sentence + ' '
            if len(test_chunk) > max_chars:
                # Salvar chunk atual e iniciar nova
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
            else:
                current_chunk += sentence + ' '
    
    # Adicionar última chunk se houver
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Garantir que nenhuma chunk excede o limite
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            # Divisão forçada como último recurso
            for i in range(0, len(chunk), max_chars - 10):
                sub_chunk = chunk[i:i + max_chars - 10].strip()
                if sub_chunk:
                    final_chunks.append(sub_chunk)
        else:
            final_chunks.append(chunk)
    
    logger.info(f"📝 Texto dividido em {len(final_chunks)} chunks (limite: {max_chars} chars)")
    for i, chunk in enumerate(final_chunks):
        logger.info(f"   Chunk {i+1}: {len(chunk)} chars - '{chunk[:50]}...'")
    
    return final_chunks

def concatenate_audio_chunks(chunk_paths: list, output_path: str, crossfade_duration: float = 0.1) -> bool:
    """Concatenar chunks de áudio com crossfade suave"""
    try:
        if not chunk_paths:
            return False
        
        if len(chunk_paths) == 1:
            # Se apenas um chunk, copiar diretamente
            audio_data, sr = sf.read(chunk_paths[0])
            sf.write(output_path, audio_data, sr)
            return True
        
        # Carregar primeiro chunk
        final_audio, sr = sf.read(chunk_paths[0])
        
        crossfade_samples = int(crossfade_duration * sr)
        
        for chunk_path in chunk_paths[1:]:
            next_audio, next_sr = sf.read(chunk_path)
            
            if next_sr != sr:
                logger.warning(f"⚠️ Sample rates diferentes: {sr} vs {next_sr}")
                continue
            
            # Aplicar crossfade entre chunks
            if len(final_audio) > crossfade_samples and len(next_audio) > crossfade_samples:
                # Fade out no final do áudio atual
                fade_out = np.linspace(1, 0, crossfade_samples)
                final_audio[-crossfade_samples:] *= fade_out
                
                # Fade in no início do próximo áudio
                fade_in = np.linspace(0, 1, crossfade_samples)
                next_audio[:crossfade_samples] *= fade_in
                
                # Sobrepor as regiões de crossfade
                overlap_region = final_audio[-crossfade_samples:] + next_audio[:crossfade_samples]
                
                # Concatenar: áudio atual (sem final) + overlap + próximo áudio (sem início)
                final_audio = np.concatenate([
                    final_audio[:-crossfade_samples],
                    overlap_region,
                    next_audio[crossfade_samples:]
                ])
            else:
                # Se chunks muito pequenos, concatenar diretamente
                final_audio = np.concatenate([final_audio, next_audio])
        
        sf.write(output_path, final_audio, sr)
        logger.info(f"🔗 {len(chunk_paths)} chunks concatenados com crossfade")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao concatenar chunks: {e}")
        return False

# ========== FUNÇÕES ORIGINAIS DE CORREÇÃO (MANTIDAS) ==========

def fix_audio_hoarseness(audio_path: str, output_path: str) -> bool:
    """Corrigir rouquidão no áudio"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Normalizar volume para evitar saturação
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Normalizar para 85% do máximo para evitar clipping
            audio_data = audio_data * (0.85 / max_val)
        
        # Aplicar filtro passa-alta suave para reduzir ruído baixo
        from scipy import signal
        
        # Filtro passa-alta em 80Hz para remover ruído de fundo
        nyquist = sr / 2
        low_cutoff = 80 / nyquist
        
        if low_cutoff < 0.99:  # Verificar se frequência é válida
            b, a = signal.butter(2, low_cutoff, btype='high')
            audio_data = signal.filtfilt(b, a, audio_data)
        
        # Suavizar picos que causam rouquidão
        # Detecção de picos excessivos
        threshold = np.std(audio_data) * 3
        peaks = np.abs(audio_data) > threshold
        
        if np.any(peaks):
            # Suavizar apenas os picos detectados
            window_size = int(0.005 * sr)  # 5ms de janela
            for i in np.where(peaks)[0]:
                start = max(0, i - window_size // 2)
                end = min(len(audio_data), i + window_size // 2)
                if end > start:
                    # Aplicar suavização gaussiana na região do pico
                    audio_data[start:end] = signal.savgol_filter(
                        audio_data[start:end], 
                        min(11, end - start) if (end - start) % 2 == 1 else min(10, end - start - 1), 
                        3
                    )
        
        # Compressão suave para uniformizar dinâmica
        # Compressor simples: reduzir apenas volumes muito altos
        compression_threshold = 0.7
        compression_ratio = 0.6
        
        mask = np.abs(audio_data) > compression_threshold
        audio_data[mask] = np.sign(audio_data[mask]) * (
            compression_threshold + 
            (np.abs(audio_data[mask]) - compression_threshold) * compression_ratio
        )
        
        sf.write(output_path, audio_data, sr)
        logger.info("🎛️ Correção de rouquidão aplicada")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao corrigir rouquidão: {e}")
        return False

# ========== APP SETUP ==========

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

# ========== ENDPOINT OTIMIZADO (RECOMENDAÇÕES CHATGPT) ==========

@app.post("/generate")
async def generate_audio_v2(
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência"),
    language: str = Form("pt", description="Código do idioma"),
    # Parâmetros otimizados para qualidade/estabilidade
    temperature: float = Form(0.75, description="Temperatura (0.65-0.85)"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(8.0, description="Penalidade de repetição"),
    top_k: int = Form(20, description="Top-k sampling"),
    top_p: float = Form(0.70, description="Top-p sampling"),
    speed: float = Form(1.0, description="Velocidade da fala"),
    remove_silence: bool = Form(True, description="Remover silêncios excessivos"),
):
    if not tts_model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Validações básicas
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Texto muito longo")
    
    # Validar formato do arquivo
    allowed_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    if not reference_audio.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato não suportado. Use: {', '.join(allowed_extensions)}"
        )
    
    # Validar parâmetros otimizados
    if not (0.65 <= temperature <= 0.85):
        raise HTTPException(status_code=400, detail="Temperatura deve estar entre 0.65-0.85")
    
    if not (0.6 <= top_p <= 0.8):
        raise HTTPException(status_code=400, detail="Top-p deve estar entre 0.6-0.8")
    
    if not (10 <= top_k <= 40):
        raise HTTPException(status_code=400, detail="Top-k deve estar entre 10-40")
    
    start_time = time.time()
    ref_path = None
    processed_ref_path = None
    output_path = None
    final_path = None
    
    try:
        logger.info(f"🎵 TTS v2 Otimizado - Texto: {len(text)} chars, Idioma: {language}")
        
        # 1. Salvar áudio de referência original
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        # 2. Pré-processar áudio de referência para 16kHz mono
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_ref_file:
            processed_ref_path = processed_ref_file.name
        
        if not preprocess_reference_audio(ref_path, processed_ref_path):
            raise HTTPException(status_code=400, detail="Falha no pré-processamento do áudio de referência")
        
        # 3. Otimizar texto para síntese
        optimized_text = optimize_text_for_speech(text, language)
        
        # 4. Dividir texto em chunks se necessário
        char_limits = get_character_limits()
        char_limit = char_limits.get(language, 160)
        
        if len(optimized_text) > char_limit:
            text_chunks = smart_text_split(optimized_text, language, char_limit)
        else:
            text_chunks = [optimized_text]
        
        logger.info(f"📝 Texto dividido em {len(text_chunks)} chunks")
        
        # 5. Gerar áudio para cada chunk
        chunk_paths = []
        
        for i, chunk_text in enumerate(text_chunks):
            logger.info(f"🎤 Processando chunk {i+1}/{len(text_chunks)}")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                chunk_path = chunk_file.name
            
            # Síntese com configurações otimizadas
            tts_model.tts_to_file(
                text=chunk_text.strip(),
                speaker_wav=processed_ref_path,  # Usa áudio pré-processado
                language=language,
                file_path=chunk_path,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
                split_sentences=True if len(chunk_text) > 160 else False  # Split automático para textos longos
            )
            
            # Verificar se arquivo foi gerado corretamente
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                logger.error(f"❌ Falha na geração do chunk {i+1}")
                continue
            
            chunk_paths.append(chunk_path)
            logger.info(f"✅ Chunk {i+1} gerado")
        
        if not chunk_paths:
            raise HTTPException(status_code=500, detail="Falha na geração de todos os chunks")
        
        # 6. Concatenar chunks se houver múltiplos
        if len(chunk_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as concat_file:
                concatenated_path = concat_file.name
            
            if concatenate_audio_chunks(chunk_paths, concatenated_path):
                # Limpar chunks individuais
                for chunk_path in chunk_paths:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
                output_path = concatenated_path
                logger.info(f"🔗 {len(chunk_paths)} chunks concatenados")
            else:
                # Se concatenação falhou, usar primeiro chunk
                output_path = chunk_paths[0]
                # Limpar outros chunks
                for chunk_path in chunk_paths[1:]:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
        else:
            output_path = chunk_paths[0]
        
        # 7. Pós-processar para 44.1kHz estéreo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as final_file:
            final_path = final_file.name
        
        if not postprocess_generated_audio(output_path, final_path):
            # Fallback: usar áudio original se pós-processamento falhar
            final_path = output_path
            logger.warning("⚠️ Usando áudio sem pós-processamento")
        
        # 8. Remover silêncios se solicitado
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as silence_file:
                silence_path = silence_file.name
            
            if remove_silence_from_audio(final_path, silence_path):
                if final_path != output_path:  # Só remove se não for o original
                    os.unlink(final_path)
                final_path = silence_path
        
        # Verificar arquivo final
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise HTTPException(status_code=500, detail="Falha na geração do áudio")
        
        # Estatísticas finais
        generation_time = time.time() - start_time
        file_size = os.path.getsize(final_path)
        
        try:
            final_audio, final_sr = sf.read(final_path)
            duration = len(final_audio) / final_sr
            logger.info(f"✅ Áudio v2 gerado em {generation_time:.2f}s - {duration:.1f}s @ {final_sr}Hz")
        except:
            logger.info(f"✅ Áudio v2 gerado em {generation_time:.2f}s - {file_size} bytes")
        
        return FileResponse(
            final_path,
            media_type="audio/wav",
            filename=f"tts_optimized_{int(time.time())}.wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-File-Size": str(file_size),
                "X-Audio-Duration": str(round(duration, 2)) if 'duration' in locals() else "unknown",
                "X-Sample-Rate": str(final_sr) if 'final_sr' in locals() else "44100",
                "X-Text-Chunks": str(len(text_chunks)),
                "X-Optimized": "true",
                "X-Version": "v2"
            },
            background=lambda: cleanup_temp_files([final_path])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na síntese v2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na síntese: {str(e)}")
    
    finally:
        # Limpar arquivos temporários (exceto o final que será enviado)
        cleanup_temp_files([ref_path, processed_ref_path, output_path])

# ========== ENDPOINT ORIGINAL (MANTIDO PARA COMPATIBILIDADE) ==========

@app.post("/generate_old")
async def generate_audio(
    text: str = Form(..., description="Texto para sintetizar"),
    reference_audio: UploadFile = File(..., description="Áudio de referência (.wav)"),
    language: str = Form("pt", description="Código do idioma"),
    temperature: float = Form(0.76, description="Temperatura (0.70-0.85) - mais baixa elimina rouquidão"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(8.0, description="Penalidade de repetição (6.0-10.0)"),
    top_k: int = Form(30, description="Top-k sampling (menor = mais estável)"),
    top_p: float = Form(0.75, description="Top-p sampling (menor = mais estável)"),
    speed: float = Form(0.96, description="Velocidade da fala (0.9-1.1)"),
    enable_text_splitting: bool = Form(True, description="Dividir texto automaticamente"),
    remove_silence: bool = Form(True, description="Remover silêncios excessivos"),
    fix_hoarseness: bool = Form(True, description="Corrigir rouquidão no áudio"),
    max_chars_per_chunk: int = Form(0, description="Máximo de caracteres por chunk (0=automático)")
):
    """Endpoint original mantido para compatibilidade - use /generate_v2 para versão otimizada"""
    if not tts_model:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Validações básicas
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
    if not (0.70 <= temperature <= 0.85):
        raise HTTPException(status_code=400, detail="Temperatura deve estar entre 0.70 e 0.85")
    
    if not (6.0 <= repetition_penalty <= 10.0):
        raise HTTPException(status_code=400, detail="Repetition penalty deve estar entre 6.0 e 10.0")
    
    if not (0.9 <= speed <= 1.1):
        raise HTTPException(status_code=400, detail="Velocidade deve estar entre 0.9 e 1.1")
    
    if not (10 <= top_k <= 50):
        raise HTTPException(status_code=400, detail="Top-k deve estar entre 10 e 50")
    
    if not (0.65 <= top_p <= 0.85):
        raise HTTPException(status_code=400, detail="Top-p deve estar entre 0.65 e 0.85")
    
    ref_path = None
    
    try:
        logger.info(f"🎵 TTS Original - Texto: {len(text)} chars, Idioma: {language}, Temp: {temperature}")
        start_time = time.time()
        
        # Verificar limite de caracteres
        char_limits = get_character_limits()
        char_limit = char_limits.get(language, 160)
        if max_chars_per_chunk > 0:
            char_limit = min(char_limit, max_chars_per_chunk)
        
        # Otimizar texto para melhor síntese
        optimized_text = optimize_text_for_speech(text, language)
        
        # Dividir texto em chunks se necessário
        if enable_text_splitting and len(optimized_text) > char_limit:
            text_chunks = smart_text_split(optimized_text, language, char_limit)
        else:
            text_chunks = [optimized_text]
        
        # Salvar áudio de referência temporariamente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            content = await reference_audio.read()
            ref_file.write(content)
            ref_path = ref_file.name
        
        # Validar áudio de referência
        validation = validate_reference_audio(ref_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Áudio de referência inválido: {validation['error']}")
        
        logger.info(f"✅ Áudio de referência válido - Duração: {validation['duration']:.1f}s")
        
        # Processar cada chunk
        chunk_paths = []
        
        for i, chunk_text in enumerate(text_chunks):
            logger.info(f"🎤 Processando chunk {i+1}/{len(text_chunks)} ({len(chunk_text)} chars)")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                chunk_path = chunk_file.name
            
            # Síntese de voz
            tts_model.tts_to_file(
                text=chunk_text,
                speaker_wav=ref_path,
                language=language,
                file_path=chunk_path,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
                split_sentences=False
            )
            
            # Verificar se arquivo foi gerado corretamente
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                logger.error(f"❌ Falha na geração do chunk {i+1}")
                continue
            
            chunk_paths.append(chunk_path)
            logger.info(f"✅ Chunk {i+1} gerado com sucesso")
        
        if not chunk_paths:
            raise HTTPException(status_code=500, detail="Falha na geração de todos os chunks")
        
        # Concatenar chunks se houver múltiplos
        if len(chunk_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as concat_file:
                concatenated_path = concat_file.name
            
            if concatenate_audio_chunks(chunk_paths, concatenated_path):
                # Limpar chunks individuais
                for chunk_path in chunk_paths:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
                current_path = concatenated_path
                logger.info(f"🔗 {len(chunk_paths)} chunks concatenados")
            else:
                # Se concatenação falhou, usar primeiro chunk
                current_path = chunk_paths[0]
                # Limpar outros chunks
                for chunk_path in chunk_paths[1:]:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
        else:
            current_path = chunk_paths[0]
        
        # Pós-processamento
        processing_steps = []
        
        # Corrigir rouquidão se solicitado
        if fix_hoarseness:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fixed_file:
                fixed_path = fixed_file.name
            
            if fix_audio_hoarseness(current_path, fixed_path):
                os.unlink(current_path)
                current_path = fixed_path
                processing_steps.append("rouquidão corrigida")
            else:
                os.unlink(fixed_path)
        
        # Remover silêncios se solicitado
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_file:
                processed_path = processed_file.name
            
            if remove_silence_from_audio(current_path, processed_path):
                os.unlink(current_path)
                current_path = processed_path
                processing_steps.append("silêncios removidos")
            else:
                os.unlink(processed_path)
        
        # Verificar arquivo final
        if not os.path.exists(current_path) or os.path.getsize(current_path) == 0:
            raise HTTPException(status_code=500, detail="Falha na geração do áudio final")
        
        generation_time = time.time() - start_time
        file_size = os.path.getsize(current_path)
        
        # Informações do áudio gerado
        try:
            final_audio, final_sr = sf.read(current_path)
            final_duration = len(final_audio) / final_sr
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - {len(text_chunks)} chunks - Duração: {final_duration:.1f}s")
            if processing_steps:
                logger.info(f"🎛️ Pós-processamento: {', '.join(processing_steps)}")
        except:
            logger.info(f"✅ Áudio gerado em {generation_time:.2f}s - {len(text_chunks)} chunks - Tamanho: {file_size} bytes")
        
        return FileResponse(
            current_path,
            media_type="audio/wav",
            filename=f"generated_{int(time.time())}.wav",
            headers={
                "X-Generation-Time": str(generation_time),
                "X-File-Size": str(file_size),
                "X-Audio-Duration": str(final_duration) if 'final_duration' in locals() else "unknown",
                "X-Text-Chunks": str(len(text_chunks)),
                "X-Processing-Steps": ",".join(processing_steps) if processing_steps else "none",
                "X-Character-Limit": str(char_limit),
                "X-Hoarseness-Fixed": "true" if fix_hoarseness else "false",
                "X-Version": "original"
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
        # Limpar arquivo de referência
        if ref_path and os.path.exists(ref_path):
            try:
                os.unlink(ref_path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover arquivo de referência {ref_path}: {e}")

# ========== OUTROS ENDPOINTS ==========

@app.get("/audio-tips")
async def get_audio_tips():
    """Dicas para melhorar a qualidade do áudio"""
    return {
        "optimization_info": {
            "v2_improvements": [
                "Pré-processamento do áudio de referência para 16kHz mono (ideal para XTTS v2)",
                "Pós-processamento para 44.1kHz estéreo (qualidade universal)",
                "Sample rate correto elimina problema de voz abafada",
                "Parâmetros otimizados para estabilidade e clareza",
                "Workflow simplificado sem correções desnecessárias"
            ],
            "recommended_endpoint": "/generate_v2 (otimizado) vs /generate (original)"
        },
        "reference_audio_tips": [
            "Use áudio de 5-15 segundos de duração",
            "Voz clara e bem articulada",
            "Sem ruído de fundo ou eco",
            "Volume adequado (não muito baixo/alto)",
            "Emoção/energia desejada na voz",
            "Qualquer formato será convertido para 16kHz mono automaticamente (v2)"
        ],
        "text_tips": [
            "Use pontuação adequada (. ! ?)",
            "Evite texto muito longo (máx 1000 chars)",
            "Escreva números por extenso (2023 → dois mil e vinte e três)",
            "Evite abreviações (Dr → Doutor)",
            "Use frases completas"
        ],
        "parameter_tips_v2": {
            "temperature": "0.65-0.85 (padrão 0.75) - controlado para estabilidade",
            "repetition_penalty": "8.0 (padrão) - evita repetições",
            "speed": "1.0 (padrão) - velocidade natural",
            "top_k": "20 (otimizado) - mais conservador que original",
            "top_p": "0.70 (otimizado) - melhor estabilidade"
        },
        "sample_rate_fix": {
            "problem": "XTTS v2 gera em 24kHz, mas se reproduzido como 44.1kHz soa abafado",
            "solution": "v2 converte automaticamente para 44.1kHz estéreo",
            "benefit": "Som claro e compatível com todos os players"
        },
        "character_limits": get_character_limits(),
        "endpoints": {
            "/generate_v2": "Versão otimizada com correções de sample rate",
            "/generate": "Versão original (mantida para compatibilidade)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )