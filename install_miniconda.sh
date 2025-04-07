#!/bin/bash
set -euo pipefail

# Production-ready script to install Miniconda on Ubuntu EC2 instances.
# The script downloads the latest Miniconda installer, installs it in /opt/miniconda3,
# configures system-wide environment variables, and updates conda.
#
# Requirements:
#   - Run as root (sudo)
#   - Internet connectivity

INSTALL_DIR="/opt/miniconda3"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_PATH="/tmp/Miniconda3-latest-Linux-x86_64.sh"
PROFILE_SCRIPT="/etc/profile.d/miniconda.sh"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Ensure the script is run as root.
if [[ "$EUID" -ne 0 ]]; then
    log "ERROR: This script must be run as root. Please run with sudo."
    exit 1
fi

# Check if Miniconda is already installed.
if [ -x "${INSTALL_DIR}/bin/conda" ]; then
    log "Miniconda is already installed at ${INSTALL_DIR}."
    exit 0
fi

log "Updating package lists..."
apt-get update

log "Downloading Miniconda installer..."
wget -O "${INSTALLER_PATH}" "${INSTALLER_URL}"

log "Installing Miniconda to ${INSTALL_DIR} in silent mode..."
bash "${INSTALLER_PATH}" -b -p "${INSTALL_DIR}"

log "Initializing conda for bash..."
"${INSTALL_DIR}/bin/conda" init bash

log "Updating conda to the latest version..."
"${INSTALL_DIR}/bin/conda" update -n base -c defaults conda -y

# Create a profile script to set the PATH system-wide.
log "Creating system-wide environment variables at ${PROFILE_SCRIPT}..."
cat <<EOF > "${PROFILE_SCRIPT}"
# Miniconda environment variables
export PATH="${INSTALL_DIR}/bin:\$PATH"
EOF
chmod +x "${PROFILE_SCRIPT}"

log "Miniconda installation is complete."
log "To use conda, either log out and log back in or source ${PROFILE_SCRIPT}:"
log "    source ${PROFILE_SCRIPT}"