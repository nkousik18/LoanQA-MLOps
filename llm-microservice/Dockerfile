FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy ONLY requirements first (good caching)
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app ./app

# Expose FastAPI port
EXPOSE 8001

# Start server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8001"]
