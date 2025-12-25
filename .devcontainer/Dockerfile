# Use a mirror for the base image to avoid connection issues in China
# FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
FROM docker.m.daocloud.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Use Tsinghua mirrors for apt (often more stable than Aliyun)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Install system dependencies
# Retry installation once if it fails due to network issues
RUN apt-get update && (apt-get install -y --fix-missing \
    git \
    curl \
    wget \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6 \
    || apt-get install -y --fix-missing \
    git \
    curl \
    wget \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6) \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --fix-missing python3.12 python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Use Tsinghua mirrors for pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install PyTorch 2.7.0 with CUDA 12.6 support
# Note: Using the command from README
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /workspace

# Copy project files (optional, usually mounted via devcontainer.json)
# COPY . /workspace

# Install project in editable mode
# This will be done in postCreateCommand to ensure the volume is mounted
