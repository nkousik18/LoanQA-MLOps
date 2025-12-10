#!/bin/bash
set -e

PROJECT_ID="mlops-loandoc-qa"
REGION="us-east1"
REPO="loan-llm-ms"
IMAGE="llm-microservice"
TAG="latest"

FULL_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"

echo "========================================="
echo " 🔧 Setting up Docker Buildx builder"
echo "========================================="

# Create builder if not exists
docker buildx inspect llm_builder >/dev/null 2>&1 || \
    docker buildx create --name llm_builder --use --driver docker-container

docker buildx use llm_builder
docker buildx inspect --bootstrap

echo "========================================="
echo " 🔐 Configuring Artifact Registry Auth"
echo "========================================="
gcloud auth configure-docker $REGION-docker.pkg.dev -q

echo "========================================="
echo " 📦 Building Image for linux/amd64"
echo "========================================="
docker buildx build \
  --builder llm_builder \
  --platform linux/amd64 \
  -t $FULL_IMAGE \
  --push \
  .

echo "========================================="
echo " ✅ Build + Push Complete!"
echo " 📌 Image URL:"
echo "     $FULL_IMAGE"
echo "========================================="

echo "🔍 Verifying image exists in Artifact Registry..."
gcloud artifacts docker images list $REPO --project=$PROJECT_ID --include-tags --filter="TAG:$TAG"

