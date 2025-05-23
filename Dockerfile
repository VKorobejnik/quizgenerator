# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Install dependencies efficiently
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Run Streamlit on the port specified by Koyeb ($PORT)
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0