FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Один вызов pip = один резолвер. torch CPU нужен только конвертеру CT2.
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "torch==2.5.1+cpu" \
    "ctranslate2==4.5.0" \
    "transformers==4.46.3" \
    "huggingface_hub==0.26.5" \
    sentencepiece \
    protobuf \
    "numpy<2" \
    "aiohttp>=3.9" \
    orjson \
    nvidia-ml-py

RUN ldconfig -p | grep -q "libcudnn.so.9" \
    && python3 -c "import ctranslate2, torch; assert ctranslate2.__version__=='4.5.0'; print('ok ct2', ctranslate2.__version__, 'torch', torch.__version__)"

COPY translator.py .

EXPOSE 8003

HEALTHCHECK --interval=15s --timeout=5s --retries=120 --start-period=120s \
    CMD curl -sf http://localhost:8003/health || exit 1

CMD ["python3", "translator.py"]