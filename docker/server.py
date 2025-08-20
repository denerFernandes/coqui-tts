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

def get_character_limits() -> dict:
    """Limites de caracteres por idioma para XTTS"""
    return {
        "pt": 200,  # Português - limite conservador para melhor qualidade
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
        max_chars = limits.get(language, 200)
    
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

def improve_audio_start_simple(audio_path: str, output_path: str) -> bool:
    """Melhoria simples do início do áudio (quando não há artefatos graves)"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Aplicar fade-in suave apenas
        fade_duration = 0.1  # 100ms
        fade_samples = int(fade_duration * sr)
        fade_samples = min(fade_samples, len(audio_data) // 4)  # Máximo 25% do áudio
        
        if fade_samples > 0:
            fade_curve = np.linspace(0, 1, fade_samples) ** 1.5  # Curva suave
            audio_data[:fade_samples] *= fade_curve
        
        sf.write(output_path, audio_data, sr)
        logger.info("🎛️ Fade-in suave aplicado ao início")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao aplicar fade-in: {e}")
        return False

def detect_and_fix_audio_artifacts(audio_path: str, output_path: str) -> bool:
    """Detectar e corrigir artefatos no início do áudio (clicks, pops, barulhos)"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Parâmetros para detecção de artefatos
        initial_duration = 1.0  # Primeiros 1 segundo
        initial_samples = int(initial_duration * sr)
        
        if len(audio_data) < initial_samples:
            initial_samples = len(audio_data)
        
        # Analisar o início do áudio
        start_segment = audio_data[:initial_samples].copy()
        
        # 1. Detectar clicks/pops (mudanças bruscas de amplitude)
        diff = np.diff(start_segment)
        click_threshold = np.std(diff) * 5  # 5x o desvio padrão
        clicks = np.abs(diff) > click_threshold
        
        if np.any(clicks):
            logger.info(f"🔧 Detectados {np.sum(clicks)} clicks/pops no início")
            
            # Suavizar clicks detectados
            click_indices = np.where(clicks)[0]
            for idx in click_indices:
                # Suavizar numa janela pequena ao redor do click
                window_start = max(0, idx - 5)
                window_end = min(len(start_segment), idx + 6)
                
                if window_end > window_start + 1:
                    # Interpolação linear para suavizar
                    start_val = start_segment[window_start] if window_start > 0 else 0
                    end_val = start_segment[window_end-1] if window_end < len(start_segment) else start_segment[-1]
                    
                    # Interpolação suave
                    window_size = window_end - window_start
                    interpolated = np.linspace(start_val, end_val, window_size)
                    start_segment[window_start:window_end] = interpolated
        
        # 2. Aplicar fade-in muito suave e longo
        fade_duration = 0.3  # 300ms de fade-in
        fade_samples = int(fade_duration * sr)
        fade_samples = min(fade_samples, len(start_segment))
        
        if fade_samples > 0:
            # Curva de fade-in suave (não linear)
            fade_curve = np.linspace(0, 1, fade_samples) ** 2  # Curva quadrática suave
            start_segment[:fade_samples] *= fade_curve
        
        # 3. Remover DC offset (componente contínua que pode causar pop)
        dc_offset = np.mean(start_segment[:min(1000, len(start_segment))])
        if abs(dc_offset) > 0.01:
            start_segment -= dc_offset
            logger.info(f"🔧 DC offset removido: {dc_offset:.4f}")
        
        # 4. Aplicar filtro passa-alta suave para remover ruído baixo
        from scipy import signal
        
        nyquist = sr / 2
        low_cutoff = 40 / nyquist  # Cortar frequências abaixo de 40Hz
        
        if low_cutoff < 0.99:
            b, a = signal.butter(1, low_cutoff, btype='high')  # Filtro suave (ordem 1)
            start_segment = signal.filtfilt(b, a, start_segment)
        
        # 5. Normalizar suavemente
        max_val = np.max(np.abs(start_segment))
        if max_val > 0:
            # Normalizar para 90% para evitar saturação
            start_segment = start_segment * (0.90 / max_val)
        
        # Reconstruir o áudio completo
        cleaned_audio = audio_data.copy()
        cleaned_audio[:initial_samples] = start_segment
        
        # Aplicar crossfade entre o início limpo e o resto do áudio
        if len(audio_data) > initial_samples:
            crossfade_duration = 0.1  # 100ms de crossfade
            crossfade_samples = int(crossfade_duration * sr)
            crossfade_samples = min(crossfade_samples, initial_samples // 2)
            
            if crossfade_samples > 0:
                # Região de crossfade
                fade_start = initial_samples - crossfade_samples
                fade_end = initial_samples
                
                # Fade out do início limpo
                fade_out = np.linspace(1, 0, crossfade_samples)
                cleaned_audio[fade_start:fade_end] *= fade_out
                
                # Fade in do áudio original
                fade_in = np.linspace(0, 1, crossfade_samples)
                original_section = audio_data[fade_start:fade_end] * fade_in
                
                # Misturar as duas seções
                cleaned_audio[fade_start:fade_end] += original_section
        
        sf.write(output_path, cleaned_audio, sr)
        logger.info("🔧 Artefatos do início removidos com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao corrigir artefatos: {e}")
        return False

def trim_warmup_from_audio(audio_path: str, output_path: str, warmup_word_count: int, sample_rate: int = 22050) -> bool:
    """Remover o aquecimento do início do áudio com maior precisão"""
    try:
        audio_data, sr = sf.read(audio_path)
        
        # Estimativa melhorada da duração do aquecimento
        # Considerando velocidade de fala natural em português
        avg_chars_per_word = 5  # Média de caracteres por palavra em português
        estimated_chars = warmup_word_count * avg_chars_per_word
        
        # Velocidade de fala: ~150 palavras por minuto = 2.5 palavras/seg
        speech_rate = 2.5  # palavras por segundo
        estimated_duration = warmup_word_count / speech_rate
        
        # Adicionar margem de segurança
        safety_margin = 0.2  # 200ms de margem
        total_duration = estimated_duration + safety_margin
        
        warmup_samples = int(total_duration * sr)
        
        logger.info(f"🎬 Removendo aquecimento: {warmup_word_count} palavras (~{total_duration:.2f}s)")
        
        if warmup_samples < len(audio_data):
            # Encontrar ponto de corte mais preciso baseado em energia
            # Analisar energia do áudio para encontrar melhor ponto de corte
            window_size = int(0.05 * sr)  # Janela de 50ms
            energy_profile = []
            
            start_analysis = max(0, warmup_samples - int(0.5 * sr))  # 500ms antes da estimativa
            end_analysis = min(len(audio_data), warmup_samples + int(0.5 * sr))  # 500ms depois
            
            for i in range(start_analysis, end_analysis, window_size):
                window = audio_data[i:i + window_size]
                energy = np.sum(window ** 2)
                energy_profile.append((i, energy))
            
            if energy_profile:
                # Encontrar o ponto com menor energia (mais provável ser entre palavras)
                min_energy_idx = min(energy_profile, key=lambda x: x[1])[0]
                
                # Se encontrou um ponto melhor, usar ele
                if abs(min_energy_idx - warmup_samples) < int(0.3 * sr):  # Dentro de 300ms
                    warmup_samples = min_energy_idx
                    logger.info(f"🎯 Ponto de corte ajustado baseado em energia: {min_energy_idx/sr:.2f}s")
            
            # Cortar o aquecimento
            trimmed_audio = audio_data[warmup_samples:]
            
            # Aplicar fade-in suave para evitar click
            fade_samples = int(0.05 * sr)  # 50ms fade-in
            if len(trimmed_audio) > fade_samples:
                fade_curve = np.linspace(0, 1, fade_samples) ** 1.5  # Curva suave
                trimmed_audio[:fade_samples] *= fade_curve
            
            sf.write(output_path, trimmed_audio, sr)
            logger.info(f"🎬 Aquecimento removido: {warmup_samples/sr:.2f}s cortados")
            return True
        else:
            # Se estimativa foi muito grande, fazer corte conservador
            conservative_cut = int(0.5 * sr)  # 500ms
            if conservative_cut < len(audio_data):
                trimmed_audio = audio_data[conservative_cut:]
                
                # Fade-in suave
                fade_samples = int(0.1 * sr)  # 100ms
                if len(trimmed_audio) > fade_samples:
                    fade_curve = np.linspace(0, 1, fade_samples) ** 1.5
                    trimmed_audio[:fade_samples] *= fade_curve
                
                sf.write(output_path, trimmed_audio, sr)
                logger.warning(f"⚠️ Corte conservador aplicado: {conservative_cut/sr:.2f}s")
                return True
            else:
                # Manter original se não conseguir cortar
                sf.write(output_path, audio_data, sr)
                logger.warning("⚠️ Não foi possível remover aquecimento - mantendo áudio original")
                return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao remover aquecimento: {e}")
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
    temperature: float = Form(0.82, description="Temperatura (0.75-0.90) - mais baixa para menos rouquidão"),
    length_penalty: float = Form(1.0, description="Penalidade de comprimento"),
    repetition_penalty: float = Form(6.0, description="Penalidade de repetição (4.0-8.0)"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.85, description="Top-p sampling"),
    speed: float = Form(1.0, description="Velocidade da fala (0.8-1.2)"),
    enable_text_splitting: bool = Form(True, description="Dividir texto automaticamente"),
    remove_silence: bool = Form(True, description="Remover silêncios excessivos"),
    improve_start: bool = Form(True, description="Melhorar qualidade do início do áudio"),
    use_warmup: bool = Form(True, description="Usar aquecimento para melhorar início"),
    fix_hoarseness: bool = Form(True, description="Corrigir rouquidão no áudio"),
    fix_start_artifacts: bool = Form(True, description="Corrigir barulhos/clicks no primeiro segundo"),
    max_chars_per_chunk: int = Form(0, description="Máximo de caracteres por chunk (0=automático)")
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
    
    # Validar parâmetros melhorados para evitar rouquidão
    if not (0.75 <= temperature <= 0.90):
        raise HTTPException(status_code=400, detail="Temperatura deve estar entre 0.75 e 0.90 para evitar rouquidão")
    
    if not (4.0 <= repetition_penalty <= 8.0):
        raise HTTPException(status_code=400, detail="Repetition penalty deve estar entre 4.0 e 8.0")
    
    if not (0.8 <= speed <= 1.2):
        raise HTTPException(status_code=400, detail="Velocidade deve estar entre 0.8 e 1.2")
    
    ref_path = None
    out_path = None
    
    try:
        # Log da requisição
        logger.info(f"🎵 Gerando áudio - Texto: {len(text)} chars, Idioma: {language}, Temp: {temperature}")
        start_time = time.time()
        
        # Verificar limite de caracteres
        char_limits = get_character_limits()
        char_limit = char_limits.get(language, 200)
        if max_chars_per_chunk > 0:
            char_limit = min(char_limit, max_chars_per_chunk)
        
        if len(text) > char_limit:
            logger.info(f"⚠️ Texto excede limite de {char_limit} chars para '{language}' - será dividido em chunks")
        
        # Otimizar texto para melhor síntese
        optimized_text = optimize_text_for_speech(text, language)
        if optimized_text != text:
            logger.info(f"📝 Texto otimizado para síntese")
        
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
        
        logger.info(f"📁 Áudio de referência salvo: {ref_path}")
        
        # Validar áudio de referência
        validation = validate_reference_audio(ref_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Áudio de referência inválido: {validation['error']}")
        
        logger.info(f"✅ Áudio de referência válido - Duração: {validation['duration']:.1f}s")
        if validation.get("recommendations"):
            for rec in validation["recommendations"]:
                logger.info(f"💡 {rec}")
        
        # Processar cada chunk
        chunk_paths = []
        
        for i, chunk_text in enumerate(text_chunks):
            logger.info(f"🎤 Processando chunk {i+1}/{len(text_chunks)} ({len(chunk_text)} chars)")
            
            # Adicionar aquecimento se habilitado e for o primeiro chunk
            warmup_word_count = 0
            if use_warmup and i == 0:  # Aquecimento apenas no primeiro chunk
                warmed_text, warmup_word_count = add_warmup_context(chunk_text, language)
                logger.info(f"🔥 Aquecimento adicionado ao primeiro chunk - {warmup_word_count} palavras")
                final_chunk_text = warmed_text
            else:
                final_chunk_text = chunk_text
            
            # Gerar áudio para este chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                chunk_path = chunk_file.name
            
            # Síntese de voz com parâmetros otimizados para evitar rouquidão
            try:
                logger.info(f"🎤 Sintetizando chunk {i+1} com parâmetros anti-rouquidão...")
                
                tts_model.tts_to_file(
                    text=final_chunk_text,
                    speaker_wav=ref_path,
                    language=language,
                    file_path=chunk_path,
                    temperature=temperature,  # Temperatura mais baixa para menos rouquidão
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                    speed=speed,
                    split_sentences=False  # Já dividimos manualmente
                )
                
            except AttributeError as e:
                if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                    logger.error("❌ Erro de compatibilidade detectado!")
                    raise HTTPException(
                        status_code=500, 
                        detail="Erro de compatibilidade: Atualize para coqui-tts>=0.27.0 e transformers<=4.46.2"
                    )
                raise
            
            # Pós-processamento do chunk
            current_chunk_path = chunk_path
            
            # Remover aquecimento se foi usado (apenas no primeiro chunk)
            if use_warmup and i == 0 and warmup_word_count > 0:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as trimmed_file:
                    trimmed_path = trimmed_file.name
                
                try:
                    audio_data, sr = sf.read(current_chunk_path)
                    if trim_warmup_from_audio(current_chunk_path, trimmed_path, warmup_word_count, sr):
                        os.unlink(current_chunk_path)
                        current_chunk_path = trimmed_path
                        logger.info(f"🎬 Aquecimento removido do chunk {i+1}")
                    else:
                        os.unlink(trimmed_path)
                except Exception as e:
                    logger.warning(f"⚠️ Falha ao remover aquecimento do chunk {i+1}: {e}")
                    os.unlink(trimmed_path)
            
            # Verificar se arquivo foi gerado corretamente
            if not os.path.exists(current_chunk_path) or os.path.getsize(current_chunk_path) == 0:
                logger.error(f"❌ Falha na geração do chunk {i+1}")
                continue
            
            chunk_paths.append(current_chunk_path)
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
        
        # Pós-processamento final
        processing_steps = []
        
        # Corrigir artefatos no início (clicks, pops, barulhos) - PRIORIDADE MÁXIMA
        if fix_start_artifacts:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as artifacts_file:
                artifacts_path = artifacts_file.name
            
            if detect_and_fix_audio_artifacts(current_path, artifacts_path):
                os.unlink(current_path)
                current_path = artifacts_path
                processing_steps.append("artefatos início corrigidos")
            else:
                os.unlink(artifacts_path)
        
        # Melhorar início do áudio (apenas se não foi processado por chunks e não aplicou correção de artefatos)
        if improve_start and len(text_chunks) == 1 and not fix_start_artifacts:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as improved_file:
                improved_path = improved_file.name
            
            # Função simplificada para quando não precisa de correção completa de artefatos
            if improve_audio_start_simple(current_path, improved_path):
                os.unlink(current_path)
                current_path = improved_path
                processing_steps.append("início melhorado")
            else:
                os.unlink(improved_path)
        
        # Corrigir rouquidão
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
                "X-Start-Artifacts-Fixed": "true" if fix_start_artifacts else "false"
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
    
    # Definir presets otimizados contra rouquidão e artefatos
    presets = {
        "energetic": {
            "temperature": 0.83,
            "repetition_penalty": 6.5,
            "speed": 1.05,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "balanced": {
            "temperature": 0.82,
            "repetition_penalty": 6.0,
            "speed": 1.0,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "calm": {
            "temperature": 0.78,
            "repetition_penalty": 5.5,
            "speed": 0.95,
            "remove_silence": False,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": False,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "expressive": {
            "temperature": 0.85,
            "repetition_penalty": 7.0,
            "speed": 1.02,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "professional": {
            "temperature": 0.80,
            "repetition_penalty": 6.5,
            "speed": 1.0,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "clear": {
            "temperature": 0.78,
            "repetition_penalty": 7.5,
            "speed": 0.98,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": True,
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
        },
        "clean": {
            "temperature": 0.76,
            "repetition_penalty": 8.0,
            "speed": 0.97,
            "remove_silence": True,
            "enable_text_splitting": True,
            "improve_start": False,  # Não precisa pois fix_start_artifacts já resolve
            "use_warmup": True,
            "fix_hoarseness": True,
            "fix_start_artifacts": True
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
        use_warmup=config["use_warmup"],
        fix_hoarseness=config["fix_hoarseness"],
        fix_start_artifacts=config["fix_start_artifacts"],
        max_chars_per_chunk=0
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
            "energetic": "Voz animada sem rouquidão nem artefatos",
            "balanced": "Equilíbrio perfeito com clareza máxima", 
            "calm": "Voz suave e cristalina",
            "expressive": "Máxima expressividade mantendo clareza",
            "professional": "Ideal para apresentações corporativas",
            "clear": "Focado em máxima clareza e eliminação de rouquidão",
            "clean": "Máxima limpeza - elimina todos os artefatos no início"
        },
        "start_artifacts_tips": [
            "Use fix_start_artifacts=true para eliminar clicks/pops no primeiro segundo",
            "Preset 'clean' é especificamente otimizado para início limpo",
            "Sistema detecta automaticamente clicks, pops e DC offset",
            "Aplica fade-in suave e filtros específicos",
            "Remove barulhos digitais comuns no início do áudio"
        ],
        "audio_quality_tips": [
            "Áudio de referência limpo é fundamental",
            "Evite áudios de referência com clicks ou pops",
            "Use áudios gravados em ambiente silencioso",
            "Normalize o volume do áudio de referência",
            "Prefira formato WAV com 22kHz ou 44kHz"
        ],
        "character_limits": get_character_limits(),
        "text_length_tips": [
            "Português: máximo 200 caracteres por chunk",
            "Inglês: máximo 250 caracteres por chunk", 
            "Sistema divide automaticamente textos longos",
            "Chunks são concatenados com crossfade suave",
            "Divisão inteligente respeita pontuação"
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