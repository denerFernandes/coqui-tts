// config.js
require('dotenv').config();

module.exports = {
  vastai: {
    apiKey: process.env.VASTAI_API_KEY,
    apiUrl: process.env.VASTAI_API_URL,
    searchParams: {
      verified: true,
      order: [['score', 'desc']],
      type: 'bid',
      allocated: false,
      gpu_name: 'RTX 3090',
      num_gpus: 1,
      min_download: parseInt(process.env.MIN_DOWNLOAD_SPEED) || 100,
      min_upload: 50,
      max_price: parseFloat(process.env.MAX_PRICE_PER_HOUR) || 0.4
    },
    dockerImage: process.env.DOCKER_IMAGE,
    instanceTimeout: parseInt(process.env.INSTANCE_TIMEOUT) || 300
  },
  coqui: {
    port: parseInt(process.env.COQUI_PORT) || 8000,
    healthCheckTimeout: parseInt(process.env.HEALTH_CHECK_TIMEOUT) || 120,
    healthCheckInterval: 5000,
    requestTimeout: 60000
  },
  paths: {
    textos: './data/textos.json',
    referencias: './data/referencias/',
    output: './data/output/'
  }
};