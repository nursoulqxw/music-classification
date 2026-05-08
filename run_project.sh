#!/usr/bin/env bash
set -e

# Ensure the virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run ./setup_venv.sh first."
  exit 1
fi

source .venv/bin/activate
python src/train.py
python src/predict.py

echo "Training and prediction completed."
