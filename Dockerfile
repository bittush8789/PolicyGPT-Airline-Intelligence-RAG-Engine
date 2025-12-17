FROM python:3.11.9-slim

# Install system dependencies for document parsing (PDF, HTML, etc.)
# libmagic-dev: File type detection (optional, but lightweight)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
