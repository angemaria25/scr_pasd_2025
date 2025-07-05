FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Ray and ML
RUN apt-get update && apt-get install -y build-essential curl netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Instalar TODAS las dependencias de Python en un solo paso
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for API, Ray, and Streamlit
EXPOSE 8000 8265 10001 6379 8501

ENV PYTHONUNBUFFERED=1
