FROM nvcr.io/nvidia/tensorrt:24.05-py3
WORKDIR /app
RUN apt-get update -y && apt install wget ffmpeg libsm6 libxext6 -y && mkdir models
RUN wget -qO- https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY pyproject.toml .
ENV PATH=$PATH:/root/.local/bin/:/app/.venv/bin/
ENV PYTHONPATH=/app/.venv/bin/python3
ENV UV_HTTP_TIMEOUT=120
RUN uv tool install pip
RUN uv sync
COPY . .
VOLUME [ "/app/models" ]
RUN chmod +x ./convert_models.sh
CMD ["bash", "./convert_models.sh"]
