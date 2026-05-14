import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from features.extraction import process_file

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# 1. Safely load trained artifacts
def load_artifacts():
    try:
        xgb_model = pickle.load(open(MODEL_DIR / "xgb_model.pkl", "rb"))
        svm_model = pickle.load(open(MODEL_DIR / "svm_model.pkl", "rb"))
        extra_trees_path = MODEL_DIR / "extra_trees_model.pkl"
        catboost_path = MODEL_DIR / "catboost_model.pkl"
        extra_trees_model = (
            pickle.load(open(extra_trees_path, "rb")) if extra_trees_path.exists() else None
        )
        try:
            catboost_model = (
                pickle.load(open(catboost_path, "rb")) if catboost_path.exists() else None
            )
        except ModuleNotFoundError:
            print("Warning: catboost not installed — CatBoost model skipped.")
            catboost_model = None
        pca = pickle.load(open(MODEL_DIR / "pca.pkl", "rb"))
        scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
        le = pickle.load(open(MODEL_DIR / "label_encoder.pkl", "rb"))
        return xgb_model, svm_model, extra_trees_model, catboost_model, pca, scaler, le
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Please ensure you have run train.py to generate the required model files.")
        sys.exit(1)

xgb_model, svm_model, extra_trees_model, catboost_model, pca, scaler, le = load_artifacts()

def predict_song(file_path):
    # Extract raw features from uploaded song (returns a list of dicts for each 5-sec segment)
    features_list = process_file(file_path)

    # Convert features to dataframe
    df = pd.DataFrame(features_list)

    # 2. Scale features using the RobustScaler
    X_scaled = scaler.transform(df)

    # 3. Apply PCA (Strictly for the SVM stream)
    X_pca = pca.transform(X_scaled)

    # 4. Predict probabilities for each 5-second segment
    xgb_probs = xgb_model.predict_proba(X_scaled)
    svm_probs = svm_model.predict_proba(X_pca)

    # 5. Blend probabilities with any available optional models
    weighted_probs = [(xgb_probs, 0.40), (svm_probs, 0.25)]
    if extra_trees_model is not None:
        weighted_probs.append((extra_trees_model.predict_proba(X_scaled), 0.20))
    if catboost_model is not None:
        weighted_probs.append((catboost_model.predict_proba(X_scaled), 0.15))

    total_weight = sum(weight for _, weight in weighted_probs)
    blended_probs = sum(probs * weight for probs, weight in weighted_probs) / total_weight

    # 6. Aggregate probabilities across the full track
    avg_probs = np.mean(blended_probs, axis=0)

    # 7. Final prediction
    pred_class = np.argmax(avg_probs)
    genre = le.inverse_transform([pred_class])[0]

    return genre, avg_probs


if __name__ == "__main__":
    # Default song, or use command-line argument
    song_name = sys.argv[1] if len(sys.argv) > 1 else "Queen - We Are The Champions.mp3"
    file_path = BASE_DIR / "data" / "sample_audio" / song_name

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        sample_dir = BASE_DIR / "data" / "sample_audio"
        if sample_dir.exists():
            print(f"Available songs in {sample_dir}:")
            for f in os.listdir(sample_dir):
                if f.endswith((".mp3", ".wav")):
                    print(f"  - {f}")
        sys.exit(1)

    print(f"Analyzing '{song_name}'... Please wait.")
    genre, probs = predict_song(file_path)

    print(f"\n=================================")
    print(f" Predicted Genre: {genre.upper()}")
    print(f"=================================\n")
    print("Confidence Scores (Sorted):")

    # Combine classes with probabilities and sort them highest to lowest for cleaner output
    prob_dict = {label: prob for label, prob in zip(le.classes_, probs)}
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)

    for label, prob in sorted_probs:
        # Only print genres that have at least a 1% chance
        if prob > 0.01: 
            print(f"  {label.capitalize():<10}: {prob:.4f}")
