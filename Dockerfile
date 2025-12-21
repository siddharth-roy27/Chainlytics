# Chainlytics Backend ML System - Dockerfile

# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p data/logs data/models registry/models registry/policies

# Expose port for API (if needed)
EXPOSE 8000

# Set permissions
RUN chmod +x ./entrypoint.sh

# Define entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "inference/decision_orchestrator.py"]