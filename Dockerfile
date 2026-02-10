# Use Python 3.10 Slim image (Fixes deprecation warnings)
FROM python:3.10-slim

# Install system dependencies (FFmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create user with ID 1000 (required by HF)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Create downloads directory with proper permissions
RUN mkdir -p /app/downloads && chown -R 1000:1000 /app/downloads

# Copy requirements first (for caching)
COPY --chown=1000:1000 backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (copy everything from backend to /app)
COPY --chown=1000:1000 backend/ .

# Switch to non-root user (required by HF)
USER 1000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOWNLOAD_DIR=/app/downloads

# Expose port (HF uses 7860)
EXPOSE 7860

# Run with Uvicorn directly (standard for HF Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
