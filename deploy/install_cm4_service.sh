#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="smartvision.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="$(command -v python3 || true)"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python3 not found in PATH"
  exit 1
fi

COMPONENTS="${1:-face voice currency}"
OCR_KEY="${OCR_SPACE_API_KEY:-helloworld}"

echo "[INFO] Installing ${SERVICE_NAME}"
echo "[INFO] Project directory: ${PROJECT_DIR}"
echo "[INFO] Run user: ${RUN_USER}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Components: ${COMPONENTS}"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=Smart Vision Assistance Coordinator (CM4)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=OCR_SPACE_API_KEY=${OCR_KEY}
ExecStart=${PYTHON_BIN} ${PROJECT_DIR}/main.py --module coordinated --target cm4 --components ${COMPONENTS}
Restart=always
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=20
StandardOutput=append:/var/log/smartvision.log
StandardError=append:/var/log/smartvision.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "[OK] Service installed and enabled for boot."
echo "[OK] It will auto-start after power loss/reboot."
echo "[INFO] Check status: sudo systemctl status ${SERVICE_NAME}"
echo "[INFO] Check logs:   tail -f /var/log/smartvision.log"
