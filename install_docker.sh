#!/bin/bash
set -euo pipefail

# Production-ready Docker installation script for Ubuntu EC2 instances.
# This script installs Docker CE, Docker CLI, containerd, and Docker Compose.
# It also adds the current non-root user to the "docker" group for easier usage.
#
# Prerequisites:
#   - The script must be run as root (or via sudo).
#   - It assumes an Ubuntu distribution (e.g., 20.04 or 22.04).

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check if running as root.
if [ "$(id -u)" -ne 0 ]; then
  log "ERROR: This script must be run as root. Please run with sudo."
  exit 1
fi

log "Updating package index..."
apt-get update

log "Installing prerequisite packages..."
apt-get install -y apt-transport-https ca-certificates curl software-properties-common

log "Adding Docker's official GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

log "Verifying Docker GPG key fingerprint..."
apt-key fingerprint 0EBFCD88

log "Adding Docker repository..."
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

log "Updating package index again..."
apt-get update

log "Installing Docker CE, Docker CLI, and containerd.io..."
apt-get install -y docker-ce docker-ce-cli containerd.io

log "Enabling and starting Docker service..."
systemctl enable docker
systemctl start docker

log "Docker installation completed. Docker version:"
docker --version

# Install Docker Compose
DOCKER_COMPOSE_VERSION="2.20.2"  # Adjust version as needed
log "Downloading Docker Compose version ${DOCKER_COMPOSE_VERSION}..."
curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

log "Setting executable permissions for Docker Compose..."
chmod +x /usr/local/bin/docker-compose

log "Docker Compose installation completed. Version:"
docker-compose --version

# Add the invoking (non-root) user to the docker group for ease of use.
USER_TO_ADD="${SUDO_USER:-$(whoami)}"
if id -nG "$USER_TO_ADD" | grep -qw docker; then
    log "User $USER_TO_ADD is already in the docker group."
else
    log "Adding user $USER_TO_ADD to the docker group..."
    usermod -aG docker "$USER_TO_ADD"
    log "User $USER_TO_ADD added to docker group. Please log out and log back in for the changes to take effect."
fi

log "Docker and Docker Compose installation complete."
log "You can now use your docker-compose.yaml file to manage and monitor the instance."