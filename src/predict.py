import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction import process_file

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load trained artifacts
model = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
le = pickle.load(open(MODEL_DIR / "label_encoder.pkl", "rb"))


def predict_song(file_path):

    # Extract features from uploaded song
    features_list = process_file(file_path)

    # Convert features to dataframe
    df = pd.DataFrame(features_list)

    # Scale features
    X = scaler.transform(df)

    # Predict probabilities for each segment
    probs = model.predict_proba(X)

    # Average probabilities across segments
    avg_probs = np.mean(probs, axis=0)

    # Final prediction
    pred_class = np.argmax(avg_probs)
    genre = le.inverse_transform([pred_class])[0]

    return genre, avg_probs


if __name__ == "__main__":
    import sys
    
    # Default song, or use command-line argument
    song_name = sys.argv[1] if len(sys.argv) > 1 else "SLTS_Nirvana.mp3"
    file_path = BASE_DIR / "data" / "sample_audio" / song_name

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        print(f"Available songs in {BASE_DIR / 'data' / 'sample_audio'}:")
        import os
        for f in os.listdir(BASE_DIR / "data" / "sample_audio"):
            if f.endswith(".mp3"):
                print(f"  - {f}")
        sys.exit(1)

    genre, probs = predict_song(file_path)

    print(f"\nPredicted Genre for '{song_name}': {genre}")
    print("\nProbabilities:")

    for label, prob in zip(le.classes_, probs):
        print(f"{label}: {prob:.4f}")