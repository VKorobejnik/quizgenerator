# Use a lightweight Python image
FROM python:3.9-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set memory-friendly environment variables
ENV OMP_THREAD_LIMIT=1 \ 
    TOKENIZERS_PARALLELISM=false

# Limits Tesseract memory usage

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0