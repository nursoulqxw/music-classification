#!/usr/bin/env bash
set -e

if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Run ./setup_venv.sh first."
  exit 1
fi

source venv/bin/activate
python src/training/train.py
python src/prediction/predict.py

echo "Training and prediction completed."
