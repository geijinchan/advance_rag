# API service image — slim Python 3.12 + FastAPI/uvicorn.
# The vLLM server runs as its own (GPU) container — see docker-compose.yml.
FROM python:3.12-slim

WORKDIR /srv

# System deps: tesseract for the OCR fallback in /ask-image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY corpus ./corpus

EXPOSE 3003

# Healthcheck hits the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:3003/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3003"]
