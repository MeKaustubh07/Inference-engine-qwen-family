# Linux CPU image (torch reference backend, fp32). The Metal backend needs macOS and the Apple GPU, which containers
# cannot reach: Docker on a Mac runs Linux in a VM with no Metal device. For Metal, run scripts/serve.py natively.
#   docker build -t inference-engine .
#   docker run --stop-timeout 30 -p 8000:8000 -v "$PWD/models:/app/models:ro" inference-engine
# (--stop-timeout must exceed serve.py's --drain-timeout, 25 s, or docker stop kills in-flight requests at 10 s)
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/serve.py scripts/serve.py

RUN useradd --create-home engine
USER engine
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
# weights are mounted at /app/models (not baked in: the image stays small and models can change without a rebuild)
ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
# Default: Qwen3.5-0.8B in fp32, ~3.5 GB. This image is for Linux hosts and CI with memory to spare. On an 8 GB Mac
# under Docker Desktop it is impractically slow even with a 5 GB VM (measured 2026-10-01: ready in 51 s, healthcheck
# healthy, but ~0.1 tok/s; the fp32 weights do not stay resident in the VM; docs/bench/raw/docker_2026-10-01.md).
# On a Mac, run scripts/serve.py natively: Metal, INT4 2B, ~50 tok/s per stream.
CMD ["--model", "qwen3.5-0.8b", "--backend", "cpu", "--kv-blocks", "256", "--max-batch", "4"]
