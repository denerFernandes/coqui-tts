require('dotenv').config();
const axios = require('axios');
const chalk = require('chalk');

async function debugVastAI() {
  console.log(chalk.bold.blue('🔍 Debug da API Vast.ai\n'));
  
  const apiKey = process.env.VASTAI_API_KEY;
  const apiUrl = process.env.VASTAI_API_URL;
  
  console.log(chalk.blue('📋 Configurações:'));
  console.log(chalk.cyan(`   API Key: ${apiKey ? apiKey.substring(0, 8) + '...' : 'NÃO DEFINIDA'}`));
  console.log(chalk.cyan(`   API URL: ${apiUrl}`));
  
  if (!apiKey || apiKey === 'sua_api_key_aqui') {
    console.log(chalk.red('\n❌ API Key não configurada!'));
    console.log(chalk.yellow('Vá em https://vast.ai → Account → API Key'));
    return;
  }
  
  const headers = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  };
  
  // Teste 1: Verificar API Key
  try {
    console.log(chalk.blue('\n🧪 Teste 1: Verificando API Key...'));
    
    const response = await axios.get(`${apiUrl}/instances`, {
      headers,
      timeout: 10000
    });
    
    console.log(chalk.green('✓ API Key válida!'));
    console.log(chalk.cyan(`   Instâncias ativas: ${response.data.instances?.length || 0}`));
    
  } catch (error) {
    console.log(chalk.red('❌ Erro no Teste 1:'));
    console.log(chalk.red(`   Status: ${error.response?.status}`));
    console.log(chalk.red(`   Mensagem: ${error.response?.data?.msg || error.message}`));
    
    if (error.response?.status === 401) {
      console.log(chalk.yellow('💡 API Key inválida - verifique no site da Vast.ai'));
      return;
    }
  }
  
  // Teste 2: Busca simples
  try {
    console.log(chalk.blue('\n🧪 Teste 2: Busca básica...'));
    
    const response = await axios.get(`${apiUrl}/bundles`, {
      headers,
      timeout: 10000
    });
    
    console.log(chalk.green('✓ Busca básica funcionou!'));
    console.log(chalk.cyan(`   Bundles: ${response.data.bundles?.length || 0}`));
    
  } catch (error) {
    console.log(chalk.red('❌ Erro no Teste 2:'));
    console.log(chalk.red(`   Status: ${error.response?.status}`));
    console.log(chalk.red(`   Data: ${JSON.stringify(error.response?.data, null, 2)}`));
  }
  
  // Teste 3: Busca com filtros
  try {
    console.log(chalk.blue('\n🧪 Teste 3: Busca com filtros...'));
    
    const response = await axios.post(`${apiUrl}/bundles`, {
      q: {
        verified: true,
        gpu_name: 'RTX 3090'
      }
    }, {
      headers,
      timeout: 10000
    });
    
    console.log(chalk.green('✓ Busca com filtros funcionou!'));
    console.log(chalk.cyan(`   RTX 3090 encontradas: ${response.data.bundles?.length || 0}`));
    
  } catch (error) {
    console.log(chalk.red('❌ Erro no Teste 3:'));
    console.log(chalk.red(`   Status: ${error.response?.status}`));
    console.log(chalk.red(`   Data: ${JSON.stringify(error.response?.data, null, 2)}`));
  }
}

debugVastAI().catch(console.error);
