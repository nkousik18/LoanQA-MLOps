# llm-microservice/Dockerfile

# Use Python 3.10 to match your local venv
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system tools needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy ONLY the microservice requirements first
COPY requirements.txt .

# 2. Install them (This happens inside the isolated container)
# This will NOT see or care about the root requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the microservice code
COPY . .

# Keep the container running so we can execute scripts
CMD ["tail", "-f", "/dev/null"]