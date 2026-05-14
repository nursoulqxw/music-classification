<div align="center">

# 🎵 Music Genre Classifier

**An end-to-end ML system that identifies the genre of any audio file in seconds.**

Upload a song → extract 63 audio features → run a weighted ensemble of 4 models → get the genre with confidence scores.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ensemble-FF6600?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=flat&logo=pytorch&logoColor=white)

**Genres:** Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock

</div>

---

## How It Works

Each uploaded audio file goes through a four-stage pipeline:

```
Audio File
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Feature Engineering                                 │
│  librosa splits audio into 5-sec segments            │
│  → 63 features per segment                          │
│    MFCCs (×13) · Chroma (×12) · Spectral · ZCR · RMS · Tempo  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Weighted Ensemble (soft voting)                     │
│                                                      │
│   XGBoost  ×0.40  ──┐                               │
│   SVM+PCA  ×0.25  ──┤──▶  Average across segments  │
│   ExtraTrees ×0.20 ──┤──▶  → Predicted Genre        │
│   CatBoost ×0.15  ──┘                               │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
               Genre + Confidence Scores
```

An optional **ResNet18 CNN** path (trained on mel spectrograms) is also available via the web UI.

---

## Model Performance

Track-level evaluation on a held-out 20% split. Segments from the same song are never split across train/test (group split by `track_id`) to prevent data leakage.

| Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **79.0%** | **78.7%** | **79.9%** | **79.0%** | **0.970** |
| CatBoost | 77.0% | 77.2% | 78.2% | 77.2% | 0.968 |
| Weighted Ensemble | 76.5% | 76.2% | 77.9% | 76.4% | 0.971 |
| SVM + PCA | 75.5% | 75.0% | 77.0% | 75.5% | 0.965 |
| Extra Trees | 75.5% | 74.4% | 76.7% | 74.9% | 0.962 |

> Best per-genre accuracy: **Classical** and **Jazz** (~95%+).
> Hardest genres to separate: **Rock** vs **Country** and **Disco** vs **Pop**.

---

## Project Structure

```
ml-project/
│
├── backend/                        # Web API
│   └── main.py                     # FastAPI server (port 6767)
│
├── src/
│   ├── features/
│   │   └── extraction.py           # librosa feature extraction (63 features)
│   ├── prediction/
│   │   ├── predict.py              # Weighted ensemble inference
│   │   └── predict_cnn.py          # ResNet18 CNN inference
│   ├── training/
│   │   └── train.py                # Train all 4 models + evaluate each
│   └── analysis/
│       ├── analyze_models.py       # Full model evaluation → reports/
│       └── generate_diagram.py     # Generate architecture diagram PNG
│
├── frontend/                       # Vanilla JS web UI
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── notebooks/
│   ├── 01_data_analysis.ipynb      # EDA, PCA, K-Means, XGBoost baseline
│   ├── 02_cnn_training.ipynb       # ResNet18 CNN training
│   └── 03_dataset_builder.ipynb    # Build feature CSV from raw WAV files
│
├── data/
│   ├── raw/                        # GTZAN WAV files (git-ignored)
│   ├── processed/                  # Feature CSV (git-ignored)
│   └── sample_audio/               # Test songs
│
├── models/                         # Saved .pkl / .pth artifacts
├── reports/
│   └── model_analysis/             # Confusion matrices, F1 heatmaps, ROC curves
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/nursoulqxw/music-classification.git
cd music-classification

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the web app

```bash
python backend/main.py
```

Open **http://localhost:6767**, drag and drop any `.mp3`, `.wav`, `.flac`, `.aac`, or `.ogg` file.

### 3. Train models from scratch *(optional)*

Requires `data/processed/music_5sec_features.csv` (built by `notebooks/03_dataset_builder.ipynb`).

```bash
python src/training/train.py
```

Trains all four models, prints per-model metrics side-by-side, and saves artifacts to `models/`.

### 4. Regenerate analysis reports

```bash
python src/analysis/analyze_models.py      # confusion matrices, ROC curves → reports/
python src/analysis/generate_diagram.py    # architecture diagram PNG → reports/
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/models` | Which models are loaded |
| `GET` | `/genres` | All 10 supported genres |
| `POST` | `/predict` | Upload audio → genre + confidence |
| `GET` | `/info` | Ensemble weights, feature count, model info |

### POST /predict

```bash
curl -X POST http://localhost:6767/predict \
  -F "file=@song.mp3" \
  -F "model_type=ml"
```

```json
{
  "filename": "song.mp3",
  "predicted_genre": "hiphop",
  "confidence": 0.81,
  "all_probabilities": {
    "hiphop": 0.81,
    "pop":    0.09,
    "rock":   0.04,
    "...":    "..."
  },
  "model_used": "ml"
}
```

---

## Dataset

**[GTZAN Genre Collection](https://marsyas.info/downloads/data_sets.html)** — 1,000 audio tracks (100 per genre, 30 seconds each, 22,050 Hz).

Each 30-second track is split into **six 5-second segments** → **5,985 samples × 64 columns** after feature extraction.

| Feature group | Features | Count |
|---|---|:---:|
| MFCC (mean + var) | Coefficients 1–13 | 26 |
| Chroma (mean + var) | Pitch classes 1–12 | 24 |
| Spectral (mean + var) | Centroid, Bandwidth, Rolloff, Flatness | 8 |
| ZCR (mean + var) | Zero-crossing rate | 2 |
| RMS (mean + var) | Root mean square energy | 2 |
| Tempo | BPM estimate | 1 |
| **Total** | | **63** |

---

## Tech Stack

| Component | Technology |
|---|---|
| Feature extraction | librosa, numpy |
| ML ensemble | XGBoost · scikit-learn (SVM, ExtraTrees) · CatBoost |
| CNN | PyTorch · ResNet18 · torchaudio |
| Backend | FastAPI · uvicorn |
| Frontend | HTML · CSS · Vanilla JS |
| Analysis | matplotlib · seaborn · pandas |
