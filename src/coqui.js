// src/coqui.js
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs-extra');
const chalk = require('chalk');
const path = require('path');
const config = require('../config');

class CoquiClient {
  constructor(instanceIp) {
    this.instanceIp = instanceIp;
    this.baseUrl = `http://${instanceIp}:${config.coqui.port}`;
  }

  // Verificar se servidor está funcionando
  async verificarSaude() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, {
        timeout: 5000
      });
      
      const isHealthy = response.data.status === 'healthy' && response.data.model_loaded;
      
      if (isHealthy) {
        console.log(chalk.green('✓ Coqui TTS pronto!'));
        if (response.data.gpu_info) {
          console.log(chalk.cyan(`   GPU: ${response.data.gpu_info.gpu_name}`));
          console.log(chalk.cyan(`   VRAM: ${response.data.gpu_info.gpu_memory_used}/${response.data.gpu_info.gpu_memory_total}`));
        }
      }
      
      return isHealthy;
      
    } catch (error) {
      return false;
    }
  }

  // Aguardar servidor ficar pronto
  async aguardarServidor(maxTentativas = 24) { // 2 minutos total
    console.log(chalk.blue('⏳ Aguardando Coqui TTS carregar modelo...'));
    
    for (let i = 0; i < maxTentativas; i++) {
      if (await this.verificarSaude()) {
        return true;
      }
      
      console.log(chalk.yellow(`   ${i + 1}/${maxTentativas} - Carregando modelo XTTS-v2...`));
      await new Promise(resolve => setTimeout(resolve, 5000)); // 5 segundos
    }
    
    throw new Error('Timeout: Servidor Coqui não ficou pronto');
  }

  // Gerar áudio individual
  async gerarAudio(texto, caminhoReferencia, nomeArquivo = 'audio.wav', opcoes = {}) {
    try {
      console.log(chalk.blue(`🎵 Gerando: ${nomeArquivo}`));
      
      const formData = new FormData();
      formData.append('text', texto);
      formData.append('reference_audio', fs.createReadStream(caminhoReferencia));
      formData.append('language', opcoes.language || 'pt');
      formData.append('temperature', opcoes.temperature || '0.75');
      formData.append('length_penalty', opcoes.length_penalty || '1.0');
      formData.append('repetition_penalty', opcoes.repetition_penalty || '1.0');
      formData.append('top_k', opcoes.top_k || '50');
      formData.append('top_p', opcoes.top_p || '0.85');
      
      const response = await axios.post(`${this.baseUrl}/generate`, formData, {
        headers: formData.getHeaders(),
        responseType: 'stream',
        timeout: config.coqui.requestTimeout
      });
      
      // Salvar arquivo
      const caminhoSaida = path.join(config.paths.output, nomeArquivo);
      await fs.ensureDir(path.dirname(caminhoSaida));
      
      const writer = fs.createWriteStream(caminhoSaida);
      response.data.pipe(writer);
      
      return new Promise((resolve, reject) => {
        writer.on('finish', () => {
          const generationTime = response.headers['x-generation-time'];
          console.log(chalk.green(`✓ ${nomeArquivo} (${generationTime}s)`));
          resolve(caminhoSaida);
        });
        writer.on('error', reject);
      });
      
    } catch (error) {
      console.error(chalk.red(`✗ Erro em ${nomeArquivo}:`), error.message);
      throw error;
    }
  }

  // Processar múltiplos textos
  async processarLote(textos, caminhoReferencia, opcoes = {}) {
    const resultados = [];
    const total = textos.length;
    const startTime = Date.now();
    
    console.log(chalk.blue(`📝 Processando ${total} textos...`));
    
    // Verificar se arquivo de referência existe
    if (!await fs.pathExists(caminhoReferencia)) {
      throw new Error(`Arquivo de referência não encontrado: ${caminhoReferencia}`);
    }
    
    for (let i = 0; i < textos.length; i++) {
      const texto = textos[i];
      const nomeArquivo = `audio_${String(i + 1).padStart(3, '0')}.wav`;
      
      try {
        const caminhoSaida = await this.gerarAudio(texto, caminhoReferencia, nomeArquivo, opcoes);
        
        resultados.push({
          index: i + 1,
          texto: texto.substring(0, 50) + (texto.length > 50 ? '...' : ''),
          arquivo: caminhoSaida,
          sucesso: true
        });
        
        // Progresso
        const progresso = Math.round(((i + 1) / total) * 100);
        const elapsed = (Date.now() - startTime) / 1000;
        const estimatedTotal = (elapsed / (i + 1)) * total;
        const remaining = Math.max(0, estimatedTotal - elapsed);
        
        console.log(chalk.cyan(`   📊 ${i + 1}/${total} (${progresso}%) - Restam ~${Math.round(remaining)}s`));
        
      } catch (error) {
        console.error(chalk.red(`✗ Falha no áudio ${i + 1}:`), error.message);
        resultados.push({
          index: i + 1,
          texto: texto.substring(0, 50) + (texto.length > 50 ? '...' : ''),
          erro: error.message,
          sucesso: false
        });
      }
    }
    
    const sucessos = resultados.filter(r => r.sucesso).length;
    const totalTime = (Date.now() - startTime) / 1000;
    
    console.log(chalk.green(`🎉 Concluído: ${sucessos}/${total} sucessos em ${Math.round(totalTime)}s`));
    
    return {
      resultados,
      resumo: {
        total,
        sucessos,
        falhas: total - sucessos,
        tempo_total: totalTime,
        tempo_por_audio: totalTime / total
      }
    };
  }

  // Obter estatísticas do servidor
  async obterEstatisticas() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`);
      return response.data;
    } catch (error) {
      return null;
    }
  }
}

module.exports = CoquiClient;