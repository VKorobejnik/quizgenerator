# Use a lightweight Python image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV STREAMLIT_GLOBAL_LOG_LEVEL=info
ENV STREAMLIT_SERVER_LOG_LEVEL=info
ENV OMP_THREAD_LIMIT=1
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0