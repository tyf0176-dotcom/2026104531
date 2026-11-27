FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1 AS python-base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /lab

FROM python-base AS llama-build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /src && curl -fsSL --retry 3 \
    https://codeload.github.com/ggml-org/llama.cpp/tar.gz/36d818e161c400eb856e4591be49f422fbeeea53 \
    -o /tmp/llama.tar.gz && tar -xzf /tmp/llama.tar.gz -C /src --strip-components=1 \
    && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build /build --target llama-server --parallel 2

FROM python-base AS lab
COPY docker/framework-lock.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir -r framework-lock.txt \
    && python -m pip install --no-cache-dir -c framework-lock.txt torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -c framework-lock.txt transformers==5.3.0 && python -m pip check
COPY docker/fetch_models.py /tmp/fetch_models.py
RUN HF_HUB_DISABLE_XET=1 python /tmp/fetch_models.py && rm /tmp/fetch_models.py
COPY --from=llama-build /build/bin/ /opt/llama/
COPY --from=llama-build /src/LICENSE /opt/llama/LICENSE
COPY benchmark/third_party/beir/LICENSE /models/encoder/LICENSE
ENV LD_LIBRARY_PATH=/opt/llama PYTHONPATH=/lab LAB_ENCODER=/models/encoder HF_HUB_OFFLINE=1 \
    LAB_BLAS_THREADS=1 LAB_MODEL_THREADS=4 TOKENIZERS_PARALLELISM=false
COPY README.md .
COPY data/ data/
COPY benchmark/ benchmark/
COPY indexes/ indexes/
RUN printf '#!/bin/sh\nexec python -m benchmark.runner "$@"\n' > /usr/local/bin/bench && chmod +x /usr/local/bin/bench
HEALTHCHECK --interval=3s --timeout=5s --start-period=30s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=3).close()"
CMD ["/opt/llama/llama-server", "-m", "/models/generator/qwen2.5-0.5b-instruct-q4_k_m.gguf", "--alias", "rag-lab", "-ngl", "0", "-t", "4", "-tb", "4", "-c", "2048", "-np", "1", "--host", "127.0.0.1", "--port", "8080"]
