# Hugging Face Spaces – MedicalXAI Backend
# HF Spaces requires port 7860 and runs as root (no custom user needed)

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "monai>=1.3.0" \
        "timm>=0.9.0" \
        "Pillow>=10.0.0" \
        "numpy>=1.24.0" \
        "pydantic>=2.0.0" \
        "python-multipart>=0.0.9" \
        "PyYAML>=6.0" \
        "scikit-learn>=1.3.0" \
        "scikit-image>=0.21.0" \
        "opencv-python-headless>=4.8.0" \
        "torchxrayvision>=1.0.0" \
        "psycopg2-binary>=2.9.0" \
        "bcrypt>=4.0.0" \
        "python-jose[cryptography]>=3.3.0" \
        "passlib[bcrypt]>=1.7.0" \
        "slowapi>=0.1.9"

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "src.serve.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
