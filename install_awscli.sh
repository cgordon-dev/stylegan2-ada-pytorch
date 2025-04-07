#!/bin/bash
set -euo pipefail

# Production script to install AWS CLI v2 on Ubuntu EC2 instances.
# This script:
#  - Updates the package index.
#  - Installs required packages: curl and unzip.
#  - Downloads the AWS CLI v2 installer.
#  - Unzips and runs the installer.
#  - Verifies the installation.
#  - Cleans up temporary files.
#
# Run this script as root (or with sudo).
#
# Usage: sudo ./install_awscli.sh

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Ensure the script is run as root.
if [[ "$EUID" -ne 0 ]]; then
  log "ERROR: This script must be run as root. Please run with sudo."
  exit 1
fi

log "Updating package lists..."
apt-get update

log "Installing prerequisites: curl and unzip..."
apt-get install -y curl unzip

# Define download URL and temporary paths.
AWSCLI_ZIP="/tmp/awscliv2.zip"
AWSCLI_DIR="/tmp/aws"

log "Downloading AWS CLI v2 installer..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$AWSCLI_ZIP"

log "Unzipping AWS CLI installer..."
unzip -q "$AWSCLI_ZIP" -d /tmp

log "Installing AWS CLI v2..."
/tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin

log "Verifying AWS CLI installation..."
aws --version

log "Cleaning up temporary files..."
rm -rf "$AWSCLI_ZIP" "$AWSCLI_DIR"

log "AWS CLI v2 installation completed successfully."