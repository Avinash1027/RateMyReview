# RateMyReview - production image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch first so requirements.txt sees it as satisfied
# and skips pulling the large CUDA wheels.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app ./app
COPY training ./training
COPY models ./models
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
