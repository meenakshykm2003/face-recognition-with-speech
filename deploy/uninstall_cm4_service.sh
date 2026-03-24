#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="smartvision.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

echo "[INFO] Stopping and disabling ${SERVICE_NAME}"
sudo systemctl stop "${SERVICE_NAME}" || true
sudo systemctl disable "${SERVICE_NAME}" || true

if [[ -f "${SERVICE_PATH}" ]]; then
  sudo rm -f "${SERVICE_PATH}"
fi

sudo systemctl daemon-reload

echo "[OK] Service removed."
