#!/bin/bash
set -euo pipefail

# Production script for setting up a p4d.24xlarge (NVIDIA A100) instance
# for running StyleGAN2-ADA-PyTorch.
#
# This script installs:
#   - NVIDIA drivers (using ubuntu-drivers autoinstall)
#   - CUDA Toolkit 11.0 (runfile installation, driver installation skipped)
#   - cuDNN 8.0.5 (expects file at /tmp/cudnn-11.0_8.0.5-1_amd64.deb)
#   - Required build tools and Python dependencies
#   - A Python virtual environment with PyTorch 1.7.1 (cu110)
#
# Assumptions:
#   - The system is Ubuntu (20.04/22.04)
#   - This script is run as root (via sudo)
#   - The cuDNN .deb file for CUDA 11.0 is placed in /tmp

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ "$EUID" -ne 0 ]]; then
    log "ERROR: This script must be run with sudo or as root."
    exit 1
fi

log "Updating package lists and upgrading packages..."
apt update && apt upgrade -y

log "Installing required packages..."
apt install -y build-essential dkms curl wget

log "Installing NVIDIA drivers..."
ubuntu-drivers autoinstall

log "Verifying NVIDIA driver installation..."
if ! command -v nvidia-smi &>/dev/null; then
    log "ERROR: nvidia-smi not found. Please reboot and re-run the script."
    exit 1
fi
nvidia-smi

# Install CUDA Toolkit 11.0
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.05_linux.run"
CUDA_RUNFILE="/tmp/cuda_11.0.3_450.51.05_linux.run"

if [ ! -f "$CUDA_RUNFILE" ]; then
    log "Downloading CUDA Toolkit 11.0 runfile..."
    wget -O "$CUDA_RUNFILE" "$CUDA_RUNFILE_URL"
fi

chmod +x "$CUDA_RUNFILE"

log "Installing CUDA Toolkit 11.0 (skipping driver installation)..."
"$CUDA_RUNFILE" --silent --toolkit --override

# Install cuDNN 8.0.5 for CUDA 11.0
CUDNN_DEB="/tmp/cudnn-11.0_8.0.5-1_amd64.deb"
if [ -f "$CUDNN_DEB" ]; then
    log "Installing cuDNN 8.0.5..."
    dpkg -i "$CUDNN_DEB"
else
    log "ERROR: cuDNN .deb file not found at $CUDNN_DEB. Please download and place it there."
    exit 1
fi

log "Configuring CUDA environment variables..."
cat << 'EOF' > /etc/profile.d/cuda.sh
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
chmod +x /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

log "Installing additional system dependencies for Python..."
apt install -y python3-venv python3-pip
pip3 install --upgrade pip
pip3 install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

log "Creating Python virtual environment..."
VENV_DIR="/opt/stylegan2_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

log "Installing PyTorch (with CUDA 11.0 support)..."
pip install torch==1.7.1+cu110 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

log "Verifying PyTorch GPU access..."
python - << 'EOF'
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
EOF

log "Setup for p4d.24xlarge (NVIDIA A100) completed successfully."