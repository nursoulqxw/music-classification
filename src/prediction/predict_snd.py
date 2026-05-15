"""
src/prediction/predict_snd.py
------------------------------
Inference module for the drum / percussion sound classifier.

Loads the XGBoost drum model and its preprocessing artifacts (scaler,
label encoder, feature names) from models/, then exposes predict_drum():
  1. Calls src/features/extraction_snd.py to extract features from the clip.
  2. Aligns feature order to what the model was trained on.
  3. Scales features and returns the predicted drum class + probabilities.

Related modules:
  src/features/extraction_snd.py  — computes the input features
  src/training/train_snd.py       — trained and saved the drum model artifacts
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Make `features` importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extraction_snd import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load artifacts
model = pickle.load(open(MODEL_DIR / "drum_model.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR / "scaler_snd.pkl", "rb"))
le = pickle.load(open(MODEL_DIR / "label_encoder_snd.pkl", "rb"))
feature_names = pickle.load(open(MODEL_DIR / "feature_names.pkl", "rb"))


def predict_drum(audio_path):

    # Extract features
    features = extract_features(audio_path)

    # Convert to dataframe
    df = pd.DataFrame([features])

    # Ensure correct order
    df = df[feature_names]

    # Scale
    X = scaler.transform(df)

    # Predict
    prediction = model.predict(X)[0]

    # Probabilities
    probs = model.predict_proba(X)[0]

    # Decode label
    label = le.inverse_transform([prediction])[0]

    return label, probs


if __name__ == "__main__":

    file_path = "Attack Kick 07.wv.wav" 

    sound = BASE_DIR / "data" / "sample_audio" / file_path

    label, probs = predict_drum(sound)

    print("\nPredicted Drum Sound:", label)

    print("\nProbabilities:")

    for drum_class, prob in zip(le.classes_, probs):
        print(f"{drum_class}: {prob:.4f}")