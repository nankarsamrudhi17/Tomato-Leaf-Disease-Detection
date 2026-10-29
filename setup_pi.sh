#!/bin/bash
# One-time setup for Raspberry Pi OS (Bookworm, 64-bit recommended).
# Run from the project root:   bash setup_pi.sh
set -e

echo "== Installing system packages =="
sudo apt update
sudo apt install -y python3-opencv python3-tk python3-pil.imagetk python3-numpy python3-venv python3-pip

echo "== Creating virtual environment (keeps system OpenCV/Tkinter visible) =="
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip

echo "== Installing TFLite runtime =="
pip install tflite-runtime || pip install ai-edge-litert

echo
echo "Setup finished."
echo "Run the app with:"
echo "  source venv/bin/activate"
echo "  cd raspberry_pi_app && python3 app.py"
