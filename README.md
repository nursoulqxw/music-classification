# Music Genre Classifier

Classifies the genre of an audio file using handcrafted audio features and an XGBoost model. Upload a song through the web UI and get back the predicted genre with per-class confidence scores.

## How it works

Audio is split into 5-second segments. For each segment, 57 features are extracted:

- **MFCC** — 13 coefficients × mean + variance
- **Chroma** — 12 pitch classes × mean + variance
- **Spectral** — centroid, bandwidth, rolloff, flatness (mean + variance each)
- **ZCR / RMS** — mean + variance
- **Tempo** — BPM estimate

Each segment is classified independently. Final prediction is the genre with the highest average probability across all segments.

**Model:** XGBoost (`n_estimators=500`, `max_depth=8`, `learning_rate=0.03`), trained with a track-level group split so no segments from the same song appear in both train and test sets.

**Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Expects `data/processed/music_5sec_features.csv` with a `label` column (genre name) and a `track_id` column (used for the group split). Saves three artifacts to `models/`:

- `model.pkl` — trained XGBoost classifier
- `scaler.pkl` — fitted StandardScaler
- `label_encoder.pkl` — fitted LabelEncoder

## Run the app

```bash
python backend.py
```

Opens at `http://localhost:6767`. Drag and drop any MP3, WAV, FLAC, AAC, or OGG file.

## Project structure

```
├── backend.py               FastAPI server, serves the frontend and /predict endpoint
├── src/
│   ├── feature_extraction.py   Audio segmentation and feature extraction (librosa)
│   ├── train.py                Model training script
│   └── predict.py              Prediction logic, can also be run directly
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── models/                  Saved model artifacts (git-ignored)
├── data/
│   ├── raw/music_dataset/   Raw WAV files per genre
│   ├── processed/           Extracted feature CSV
│   └── sample_audio/        A few MP3s for quick testing
└── notebooks/
    └── dataset_builder.ipynb   Builds the feature CSV from raw audio
```

## API

| Method | Path       | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/`        | Web UI                                           |
| POST   | `/predict` | Upload an audio file, returns genre + confidence |
| GET    | `/genres`  | List all genres the model knows                  |

`POST /predict` response:

```json
{
  "filename": "song.mp3",
  "predicted_genre": "hiphop",
  "confidence": 0.81,
  "all_probabilities": {
    "hiphop": 0.81,
    "pop": 0.09,
    ...
  }
}
```

## Direct prediction (no server)

```bash
python src/predict.py data/sample_audio/SLTS_Nirvana.mp3
```
