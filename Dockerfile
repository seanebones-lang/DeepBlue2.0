# DeepBlue 2.0 - Ultimate AI System Dockerfile
# Multi-stage build for maximum optimization

# Stage 1: Base Python environment
FROM python:3.11-slim as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM python-base as python-deps

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Node.js environment
FROM node:18-alpine as node-base

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production && npm cache clean --force

# Stage 4: Frontend build
FROM node:18-alpine as frontend-build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev)
RUN npm ci

# Copy frontend source
COPY frontend/ ./frontend/
COPY vite.config.ts tsconfig*.json tailwind.config.js ./

# Build frontend
RUN npm run build

# Stage 5: Production image
FROM python-base as production

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r deepblue && useradd -r -g deepblue deepblue

# Copy Python dependencies
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy application code
COPY core/ ./core/
COPY api/ ./api/
COPY requirements.txt ./

# Copy built frontend
COPY --from=frontend-build /app/dist ./static

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/chroma_db && \
    chown -R deepblue:deepblue /app

# Switch to non-root user
USER deepblue

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

