#!/bin/bash

echo "Starting LoanDocAI vLLM microservice..."

# Activate the correct venv
source ./venv/bin/activate

uvicorn app.server:app \
  --host 0.0.0.0 \
  --port 8010 \
  --reload

