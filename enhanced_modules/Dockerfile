FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PYTHON_VERSION=3.8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Create conda environment
RUN conda create -y -n stylegan python=${PYTHON_VERSION} && \
    conda clean -ya
    
ENV CONDA_DEFAULT_ENV="stylegan"
ENV CONDA_PREFIX="/root/miniconda3/envs/$CONDA_DEFAULT_ENV"
ENV PATH="$CONDA_PREFIX/bin:$PATH"

# Install PyTorch with CUDA
RUN conda install -y -c pytorch -c nvidia \
    pytorch=1.9.1 \
    torchvision \
    cudatoolkit=11.3 \
    && conda clean -ya

# Install other Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    ninja \
    scipy \
    click \
    pillow \
    matplotlib \
    requests \
    tqdm \
    pyspng \
    pyyaml \
    lpips \
    tensorboard \
    wandb

# Create directories for mounting volumes
RUN mkdir -p /workspace/datasets /workspace/results /workspace/logs /workspace/configs

# Copy StyleGAN2-ADA code
WORKDIR /workspace/stylegan2-ada-pytorch
COPY . .

# Ensure the custom ops are built
RUN pip install -e .

# Set up environment for the CLI
ENV PYTHONPATH="/workspace/stylegan2-ada-pytorch:${PYTHONPATH}"

# Create entrypoint script for easy usage
RUN echo '#!/bin/bash\n\
python enhanced_modules/stylegan_cli.py "$@"' > /usr/local/bin/stylegan_cli && \
    chmod +x /usr/local/bin/stylegan_cli

ENTRYPOINT ["stylegan_cli"]