# ── Stage 1: Build frontend ──────────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /frontend

# Install deps first (cache layer — only re-runs when package files change)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Build (Vite outputs to ../app/static i.e. /app/static in this stage)
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user for least-privilege execution
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install Python deps (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml ./
RUN pip install --no-cache-dir . && \
    pip cache purge

# Copy application source
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup main.py ./

# Overlay Vite-built frontend
COPY --from=frontend-builder --chown=appuser:appgroup /app/static ./app/static

# Prepare writable runtime dir; strip any accidental .env from image
RUN mkdir -p logs && \
    rm -f .env && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["python", "main.py"]
