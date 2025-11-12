# ============================================================
# Base image with CUDA 13
# ============================================================
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# ============================================================
# System dependencies
# ============================================================
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget git build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Set working directory
# ============================================================
WORKDIR /workspace

# ============================================================
# Copy only requirements.txt first (to preserve pip cache)
# ============================================================
COPY requirements.txt /workspace/requirements.txt

# ============================================================
# Install dependencies (cached until requirements.txt changes)
# ============================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch --extra-index-url https://download.pytorch.org/whl/cu130 \
        cudf-cu12 cuml-cu12 cupy-cuda12x \
        duckdb fastapi uvicorn streamlit aiohttp websockets \
        ruptures scikit-learn pyyaml pandas pytest requests python-dotenv PyJWT && \
    pip install --no-cache-dir arch   # 👈 תיקון חשוב
# ============================================================
# [omerGPT GPU Support Patch]
# Adds optional RAPIDS, HMM, Telegram, and validation dependencies
# ============================================================
RUN pip install --no-cache-dir \
        python-telegram-bot==20.3 \
        hmmlearn vaderSentiment && \
    echo "✅ GPU patch installed successfully"

# ============================================================
# Copy all remaining source files
# ============================================================
COPY . /workspace


# ============================================================
# Set environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# ============================================================
# Create logs directory to avoid FileNotFoundError
# ============================================================
RUN mkdir -p /workspace/logs
# ============================================================
# [Environment setup confirmation + health tools]
# ============================================================
RUN apt-get update && apt-get install -y curl && \
    echo "Container ready for omerGPT GPU execution"

# Healthcheck placeholder (optional, for docker-compose)
HEALTHCHECK CMD curl -f http://localhost:9000/health || exit 1

# ============================================================
# Default command (override by docker-compose)
# ============================================================
CMD ["bash"]

