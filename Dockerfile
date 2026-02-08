# Use Python 3.9 Slim image
FROM python:3.9-slim

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
CMD gunicorn -w 4 -b 0.0.0.0:$PORT api.index:app
