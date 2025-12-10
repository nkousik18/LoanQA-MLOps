#!/bin/bash
set -e

IMAGE="us-east1-docker.pkg.dev/mlops-loandoc-qa/loan-llm-ms/llm-microservice:latest"
GREEN="llm-ms-green"
BLUE="llm-ms-blue"
PROD_PORT=8001
GREEN_PORT=8005

echo "🔵🟢 Starting Blue/Green deployment..."

echo "📥 Pulling latest image..."
docker pull $IMAGE

# ------------------------------------------
# 1. START GREEN (staging)
# ------------------------------------------
echo "🟢 Starting GREEN container on port ${GREEN_PORT}..."
docker run -d \
  --rm \
  --gpus all \
  --name $GREEN \
  -p ${GREEN_PORT}:8000 \
  $IMAGE

# ------------------------------------------
# 2. HEALTH CHECK LOOP
# ------------------------------------------
echo "🔍 Waiting for GREEN to become healthy..."

for i in {1..30}; do
  if curl -s http://localhost:${GREEN_PORT}/health | grep -q "ok"; then
    echo "🟢 GREEN is healthy!"
    HEALTHY=1
    break
  fi
  echo "⏳ attempt $i/30"
  sleep 2
done

if [ -z "$HEALTHY" ]; then
  echo "❌ GREEN FAILED HEALTH CHECK"
  docker logs $GREEN
  docker stop $GREEN
  exit 1
fi

# ------------------------------------------
# 3. PROMOTE GREEN → BLUE (systemd controls BLUE)
# ------------------------------------------
echo "🔄 Promoting GREEN → BLUE..."

# Stop BLUE container managed by systemd
sudo systemctl stop llm-microservice

# Spin up BLUE with updated image
docker run -d \
  --rm \
  --gpus all \
  --name $BLUE \
  -p ${PROD_PORT}:8000 \
  $IMAGE

# Restart systemd (logs/monitoring)
sudo systemctl start llm-microservice

# Kill GREEN
docker stop $GREEN

echo "🎉 Deployment succeeded!"
echo "🟦 Production BLUE is live on port ${PROD_PORT}"
