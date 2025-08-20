#!/bin/bash
# scripts/build-docker.sh

set -e

echo "🐳 Construindo imagem Docker do Coqui TTS..."

# Configurações - SUBSTITUA SEU_USUARIO pelo seu username do Docker Hub
IMAGE_NAME="seu_usuario/coqui-tts-xtts-v2"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Verificar se Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está rodando. Inicie o Docker primeiro."
    exit 1
fi

# Verificar se arquivos necessários existem
if [ ! -f "docker/Dockerfile" ]; then
    echo "❌ Arquivo docker/Dockerfile não encontrado!"
    exit 1
fi

if [ ! -f "docker/requirements.txt" ]; then
    echo "❌ Arquivo docker/requirements.txt não encontrado!"
    exit 1
fi

if [ ! -f "docker/server.py" ]; then
    echo "❌ Arquivo docker/server.py não encontrado!"
    exit 1
fi

# Build da imagem
echo "📦 Fazendo build da imagem..."
echo "   Nome: $FULL_IMAGE_NAME"
echo "   Contexto: docker/"

docker build -t $FULL_IMAGE_NAME docker/

# Verificar se build foi bem-sucedido
if [ $? -eq 0 ]; then
    echo "✅ Build concluído com sucesso!"
else
    echo "❌ Erro no build da imagem!"
    exit 1
fi

# Verificar tamanho da imagem
echo ""
echo "📏 Informações da imagem:"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Testar imagem localmente (opcional)
echo ""
echo "🧪 Para testar a imagem localmente, execute:"
echo "docker run --rm -p 8000:8000 --gpus all $FULL_IMAGE_NAME"
echo ""
echo "Depois acesse: http://localhost:8000/health"

# Perguntar se quer fazer push
echo ""
read -p "🚀 Fazer push para Docker Hub? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Fazendo login no Docker Hub..."
    docker login
    
    if [ $? -eq 0 ]; then
        echo "📤 Fazendo push da imagem..."
        docker push $FULL_IMAGE_NAME
        
        if [ $? -eq 0 ]; then
            echo "✅ Imagem enviada com sucesso!"
            echo ""
            echo "📋 Configure no seu .env:"
            echo "DOCKER_IMAGE=$FULL_IMAGE_NAME"
        else
            echo "❌ Erro ao fazer push!"
            exit 1
        fi
    else
        echo "❌ Erro no login do Docker Hub!"
        exit 1
    fi
else
    echo "⏭️  Push cancelado."
    echo ""
    echo "📋 Para fazer push manualmente:"
    echo "docker login"
    echo "docker push $FULL_IMAGE_NAME"
fi

echo ""
echo "🎉 Processo concluído!"
echo ""
echo "📝 Próximos passos:"
echo "1. Configure o .env com DOCKER_IMAGE=$FULL_IMAGE_NAME"
echo "2. Configure VASTAI_API_KEY no .env"
echo "3. Prepare seus dados em data/textos.json"
echo "4. Execute: npm start"