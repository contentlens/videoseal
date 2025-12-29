FROM nvcr.io/nvidia/tensorrt:24.05-py3
RUN apt-get update -y && apt-get install wget ffmpeg libsm6 libxext6 -y
RUN wget -qO- https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY pyproject.toml .
ENV PATH=$PATH:/root/.local/bin/
ENV UV_PROJECT_ENVIRONMENT=/usr
ENV UV_HTTP_TIMEOUT=120
RUN uv sync
RUN uv pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.0.1 --system
COPY . .
VOLUME [ "/app/models" ]
CMD [
    "python", "export_embedder_trt.py" "&&", 
    "wget", "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit", "-P", "models"
]
