FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "aiohttp>=3.9.0" \
    "orjson>=3.9.0" \
    yake \
    "tiktoken>=0.5.0" \
    "langdetect>=1.0.9" \
    "requests>=2.31.0" \
    "aioprometheus>=23.0.0"

# exorde_data без рекурсивных сабмодулей — см. комментарий в bpipe/Dockerfile
RUN git clone --branch full --single-branch --no-recurse-submodules --depth 1 \
    https://github.com/exorde-labs/exorde_data.git /tmp/exorde_data \
    && pip install --no-cache-dir /tmp/exorde_data \
    && rm -rf /tmp/exorde_data

# tiktoken при первом обращении качает r50k_base.tiktoken с
# openaipublic.blob.core.windows.net. В рантайме у контейнера выхода в
# интернет нет (DNS не резолвится), поэтому кодировку скачиваем ЗАРАНЕЕ,
# при сборке образа (там сеть есть, как и для pip).
# Кэш кладём в /opt/tiktoken, а НЕ в /root/.cache: эта папка перекрывается
# volume ./models_cache и файл бы пропал.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/tiktoken \
    && python -c "import tiktoken; e = tiktoken.get_encoding('r50k_base'); assert e.encode('hello'); print('✅ tiktoken r50k_base cached')"

COPY . .

CMD ["python", "upipe.py"]
