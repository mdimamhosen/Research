# syntax=docker/dockerfile:1
# PaddleOCR-VL batch CLI — CPU image (works on lab PC / VPS without GPU).
# First run downloads models into a Docker volume (~GBs).

FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OCR_ENGINE=paddle \
    PADDLE_OCR_BACKEND=vl \
    PADDLE_OCR_DEVICE=cpu \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PaddlePaddle CPU (must match Python 3.13)
RUN pip install --no-cache-dir \
    paddlepaddle==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py app.py ./
COPY src ./src
COPY .env.example ./.env.example

RUN mkdir -p data/pdfs data/txt data/samples

VOLUME ["/app/data/pdfs", "/app/data/txt"]

ENTRYPOINT ["python", "cli.py"]
CMD ["-i", "data/pdfs", "-o", "data/txt", "--engine", "paddle", "--skip-existing"]
