# Use Python 3.11 Slim image (Fixes deprecation warnings)
FROM python:3.11-slim

# Install system dependencies (FFmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Render sets PORT env var)
EXPOSE 5000

# Run with Gunicorn
# -w 1: Single worker to save RAM (Prevent OOM on Render Free Tier)
# -t 120: 2-minute timeout for infinite video processing
# --threads 4: Handle concurrent requests without extra RAM
CMD gunicorn -w 1 --threads 4 -t 120 -b 0.0.0.0:$PORT api.index:app
