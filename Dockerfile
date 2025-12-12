# Dockerfile for Doc-Understand Streamlit App
# Deploys to Google Cloud Run

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY aws_extraction_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r aws_extraction_requirements.txt

# Pre-download Sentence Transformer model (avoids HuggingFace rate limits at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy entire application
COPY . .

# Create necessary directories
RUN mkdir -p logs/aws_extraction_logs && \
    mkdir -p data/local_pipeline/sessions && \
    mkdir -p reports/aws_extraction_reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run Streamlit
# Adjust path if streamlit file location is different
CMD streamlit run scripts/LLM/forms_llm/ui/streamlit_app_monitored.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false