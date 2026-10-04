FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU-only PyTorch for fast, tiny build)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy bot code and weights
COPY . /app/

# Hugging Face Spaces / Container port (if health check web server is run)
EXPOSE 7860

CMD ["python", "-u", "app.py"]
