FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install PyTorch 2.7.0 with CUDA 12.6 support
# Note: Using the command from README
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
RUN pip install \
    # Core dependencies
    "timm>=1.0.17" \
    "numpy==1.26" \
    tqdm \
    "ftfy==6.1.1" \
    regex \
    "iopath>=0.1.10" \
    typing_extensions \
    huggingface_hub \
    # Dev dependencies
    pytest \
    pytest-cov \
    "black==24.2.0" \
    "ufmt==2.8.0" \
    "ruff-api==0.1.0" \
    "usort==1.0.2" \
    "gitpython==3.1.31" \
    yt-dlp \
    pandas \
    opencv-python \
    pycocotools \
    numba \
    python-rapidjson \
    # Train dependencies
    hydra-core \
    submitit \
    tensorboard \
    zstandard \
    scipy \
    torchmetrics \
    fvcore \
    fairscale \
    scikit-image \
    scikit-learn \
    # Notebook dependencies
    matplotlib \
    jupyter \
    notebook \
    ipywidgets \
    ipycanvas \
    ipympl \
    decord \
    einops \
    # Tools
    ruff \
    isort \
    flake8 \
    mypy

WORKDIR /workspace

# Copy project files (optional, usually mounted via devcontainer.json)
# COPY . /workspace

# Install project in editable mode
# This will be done in postCreateCommand to ensure the volume is mounted
