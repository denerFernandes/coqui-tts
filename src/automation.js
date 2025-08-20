// src/automation.js
const fs = require('fs-extra');
const path = require('path');
const chalk = require('chalk');
const VastAI = require('./vastai');
const CoquiClient = require('./coqui');
const config = require('../config');

class AutomationManager {
  constructor() {
    this.vastai = new VastAI();
    this.coquiClient = null;
    this.instanceId = null;
    this.instanceInfo = null;
  }

  async executar() {
    console.log(chalk.bold.blue('🚀 Iniciando Automação Coqui TTS + Vast.ai\n'));
    
    try {
      // 1. Carregar dados
      const dados = await this.carregarDados();
      
      // 2. Buscar e criar instância
      await this.configurarInstancia();
      
      // 3. Processar áudios
      const resultados = await this.processarAudios(dados);
      
      // 4. Salvar relatório
      await this.salvarRelatorio(resultados);
      
      console.log(chalk.bold.green('\n✅ Automação concluída com sucesso!'));
      
    } catch (error) {
      console.error(chalk.bold.red('\n❌ Erro na automação:'), error.message);
      throw error;
    } finally {
      // 5. Limpar recursos
      await this.limparRecursos();
    }
  }

  async carregarDados() {
    console.log(chalk.blue('📄 Carregando dados...'));
    
    // Verificar se arquivo existe
    if (!await fs.pathExists(config.paths.textos)) {
      throw new Error(`Arquivo não encontrado: ${config.paths.textos}`);
    }
    
    const dados = await fs.readJson(config.paths.textos);
    
    // Validar estrutura
    if (!dados.textos || !Array.isArray(dados.textos)) {
      throw new Error('Arquivo deve ter array "textos"');
    }
    
    if (!dados.referencia) {
      throw new Error('Arquivo deve ter campo "referencia" com caminho do áudio');
    }
    
    // Verificar arquivo de referência
    const caminhoReferencia = path.resolve(dados.referencia);
    if (!await fs.pathExists(caminhoReferencia)) {
      throw new Error(`Áudio de referência não encontrado: ${caminhoReferencia}`);
    }
    
    dados.referencia = caminhoReferencia;
    
    console.log(chalk.green(`✓ ${dados.textos.length} textos carregados`));
    console.log(chalk.green(`✓ Referência: ${path.basename(dados.referencia)}`));
    
    return dados;
  }

  async configurarInstancia() {
    console.log(chalk.blue('\n🔍 Configurando instância GPU...'));
    
    // Buscar instâncias disponíveis
    const instancias = await this.vastai.buscarInstancias();
    
    if (instancias.length === 0) {
      throw new Error('Nenhuma instância adequada encontrada');
    }
    
    // Usar a primeira (mais barata)
    const melhorInstancia = instancias[0];
    console.log(chalk.cyan(`💰 Selecionada: ${melhorInstancia.gpu_name} - ${melhorInstancia.min_bid}/h`));
    
    // Criar instância
    this.instanceId = await this.vastai.criarInstancia(melhorInstancia.id);
    
    // Aguardar ficar pronta
    this.instanceInfo = await this.vastai.aguardarInstancia(this.instanceId);
    
    // Configurar cliente Coqui
    this.coquiClient = new CoquiClient(this.instanceInfo.ip);
    
    // Aguardar Coqui carregar
    await this.coquiClient.aguardarServidor();
    
    console.log(chalk.green('✓ Instância configurada e pronta!'));
  }

  async processarAudios(dados) {
    console.log(chalk.blue('\n🎵 Processando áudios...'));
    
    const opcoes = dados.opcoes || {};
    
    // Mostrar configurações
    console.log(chalk.cyan('⚙️  Configurações:'));
    console.log(chalk.cyan(`   Idioma: ${opcoes.language || 'pt'}`));
    console.log(chalk.cyan(`   Temperatura: ${opcoes.temperature || 0.75}`));
    console.log(chalk.cyan(`   Referência: ${path.basename(dados.referencia)}`));
    
    // Processar
    const resultados = await this.coquiClient.processarLote(
      dados.textos,
      dados.referencia,
      opcoes
    );
    
    return resultados;
  }

  async salvarRelatorio(resultados) {
    console.log(chalk.blue('\n📊 Salvando relatório...'));
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const nomeRelatorio = `relatorio_${timestamp}.json`;
    const caminhoRelatorio = path.join(config.paths.output, nomeRelatorio);
    
    const relatorio = {
      timestamp: new Date().toISOString(),
      instancia: {
        id: this.instanceId,
        ip: this.instanceInfo?.ip,
        custo_estimado: '$0.00' // Calcular baseado no tempo
      },
      resultados: resultados.resultados,
      resumo: resultados.resumo,
      arquivos_gerados: resultados.resultados
        .filter(r => r.sucesso)
        .map(r => path.basename(r.arquivo))
    };
    
    await fs.writeJson(caminhoRelatorio, relatorio, { spaces: 2 });
    
    console.log(chalk.green(`✓ Relatório salvo: ${nomeRelatorio}`));
    console.log(chalk.cyan(`📁 Pasta de saída: ${config.paths.output}`));
  }

  async limparRecursos() {
    if (this.instanceId) {
      console.log(chalk.blue('\n🧹 Limpando recursos...'));
      
      try {
        await this.vastai.destruirInstancia(this.instanceId);
        console.log(chalk.green('✓ Instância destruída'));
      } catch (error) {
        console.error(chalk.red('⚠️  Erro ao destruir instância:'), error.message);
        console.log(chalk.yellow(`⚠️  Destrua manualmente: Instância ${this.instanceId}`));
      }
    }
  }
}

// Executar se chamado diretamente
if (require.main === module) {
  const automation = new AutomationManager();
  
  automation.executar()
    .then(() => {
      console.log(chalk.bold.green('\n🎉 Processo concluído!'));
      process.exit(0);
    })
    .catch((error) => {
      console.error(chalk.bold.red('\n💥 Falha no processo:'), error.message);
      process.exit(1);
    });
}

module.exports = AutomationManager;