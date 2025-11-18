FROM nvcr.io/nvidia/tensorrt:25.09-py3
WORKDIR /app
RUN apt-get update -y && apt install wget ffmpeg libsm6 libxext6 -y && mkdir models
RUN wget -qO- https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY . .
ENV PATH=$PATH:/root/.local/bin/:/app/.venv/bin/
ENV PYTHONPATH=/app/.venv/bin/python3
ENV UV_HTTP_TIMEOUT=120
RUN uv sync
VOLUME [ "/app/models" ]
CMD ["python3.10", "export_detector_trt.py", "&&", "python3.10", "export_embedder_trt.py"]
