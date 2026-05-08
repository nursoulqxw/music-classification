# ML Final Project

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run training

```bash
python src/train.py
```

This will train the model and save artifacts to `models/model.pkl`, `models/scaler.pkl`, and `models/label_encoder.pkl`.

## Run prediction

```bash
python src/predict.py
```

This script loads the saved model and predicts the genre for the sample audio file located at `data/sample_audio/Dr. Dre Feat. Snoop Dogg - Still D.R.E..mp3`.

## Project structure

- `src/` — source code for model training, prediction, and feature extraction
- `data/processed/` — prepared dataset files
- `data/raw/` — raw dataset directory (`music_dataset`)
- `data/sample_audio/` — example MP3 files for prediction testing
- `models/` — trained model artifacts
- `notebooks/` — Jupyter notebook for dataset building and exploration

## Convenience scripts

- `setup_venv.sh` creates the `.venv` folder and installs requirements.
- `run_project.sh` runs `src/train.py` and then `src/predict.py`.

## Notes

- Make sure `data/processed/music_5sec_features.csv` exists.
- Use `data/sample_audio/` for example MP3 prediction files.
- If you want to test another MP3, update `file_path` in `src/predict.py` or call `predict_song()` from another script.
