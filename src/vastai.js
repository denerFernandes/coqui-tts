// src/vastai.js
const axios = require('axios');
const chalk = require('chalk');
const config = require('../config');

class VastAI {
  constructor() {
    this.apiKey = config.vastai.apiKey;
    this.apiUrl = config.vastai.apiUrl;
    this.headers = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async buscarInstancias() {
    try {
      console.log(chalk.blue('🔍 Buscando instâncias GPU disponíveis...'));
      
      const response = await axios.post(`${this.apiUrl}/bundles`, {
        q: config.vastai.searchParams
      }, { headers: this.headers });

      const instancias = response.data.bundles || [];
      
      // Filtrar melhores opções
      const melhores = instancias
        .filter(inst => inst.compute_cap >= 61) // RTX 3090 ou superior
        .filter(inst => inst.reliability >= 0.95)
        .filter(inst => inst.gpu_ram >= 20) // Mínimo 20GB VRAM
        .sort((a, b) => a.min_bid - b.min_bid)
        .slice(0, 3);

      console.log(chalk.green(`✓ ${melhores.length} instâncias adequadas encontradas`));
      
      // Mostrar opções
      melhores.forEach((inst, i) => {
        console.log(chalk.cyan(`   ${i + 1}. ${inst.gpu_name} - $${inst.min_bid}/h - ${inst.reliability*100}% confiável`));
      });
      
      return melhores;
      
    } catch (error) {
      console.error(chalk.red('✗ Erro ao buscar instâncias:'), error.message);
      throw error;
    }
  }

  async criarInstancia(bundleId) {
    try {
      console.log(chalk.blue(`🚀 Criando instância com Docker...`));

      // Comando Docker para rodar na inicialização
      const dockerCmd = `docker run -d --gpus all -p 8000:8000 --name coqui-tts ${config.vastai.dockerImage}`;
      
      const response = await axios.post(`${this.apiUrl}/asks/${bundleId}`, {
        price: config.vastai.searchParams.max_price,
        disk: 30, // Menor porque Docker cuida do ambiente
        image: 'nvidia/cuda:11.8-runtime-ubuntu20.04', // Imagem base leve
        onstart: dockerCmd,
        env: {
          'NVIDIA_VISIBLE_DEVICES': 'all'
        }
      }, { headers: this.headers });

      if (response.data.success) {
        const instanceId = response.data.new_contract;
        console.log(chalk.green(`✓ Instância criada! ID: ${instanceId}`));
        console.log(chalk.blue(`🐳 Iniciando container Docker...`));
        return instanceId;
      } else {
        throw new Error('Falha ao criar instância');
      }
    } catch (error) {
      console.error(chalk.red('✗ Erro ao criar instância:'), error.message);
      throw error;
    }
  }

  async verificarStatus(instanceId) {
    try {
      const response = await axios.get(`${this.apiUrl}/instances`, {
        headers: this.headers
      });

      const instance = response.data.instances.find(inst => inst.id == instanceId);
      
      if (instance) {
        return {
          status: instance.actual_status,
          ip: instance.public_ipaddr,
          port: instance.ssh_port,
          isReady: instance.actual_status === 'running'
        };
      }
      
      return null;
    } catch (error) {
      console.error(chalk.red('✗ Erro ao verificar status:'), error.message);
      throw error;
    }
  }

  async aguardarInstancia(instanceId, maxTentativas = 40) {
    console.log(chalk.blue('⏳ Aguardando instância + Docker ficar pronto...'));
    
    for (let i = 0; i < maxTentativas; i++) {
      const status = await this.verificarStatus(instanceId);
      
      if (status && status.isReady) {
        console.log(chalk.green('✓ Instância pronta!'));
        console.log(chalk.cyan(`   Endpoint: http://${status.ip}:8000`));
        return status;
      }
      
      console.log(chalk.yellow(`   ${i + 1}/${maxTentativas} - Status: ${status?.status || 'unknown'}`));
      await this._sleep(8000); // 8 segundos (Docker precisa de tempo)
    }
    
    throw new Error('Timeout: Instância não ficou pronta');
  }

  async destruirInstancia(instanceId) {
    try {
      console.log(chalk.blue(`🔥 Destruindo instância ${instanceId}...`));
      
      const response = await axios.delete(`${this.apiUrl}/instances/${instanceId}`, {
        headers: this.headers
      });

      if (response.data.success) {
        console.log(chalk.green('✓ Instância destruída!'));
        return true;
      }
      
      return false;
    } catch (error) {
      console.error(chalk.red('✗ Erro ao destruir instância:'), error.message);
      return false;
    }
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = VastAI;