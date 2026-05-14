"""
src/training/train_snd.py
--------------------------
Training pipeline for the drum / percussion sound classifier.

Reads data/processed/final_drum_dataset.csv, trains an XGBoost classifier
to distinguish drum types (kick, snare, hi-hat, etc.), evaluates it, and
saves four artifacts to models/:

  models/drum_model.pkl       XGBoost drum classifier
  models/scaler_snd.pkl       StandardScaler fitted on drum features
  models/label_encoder_snd.pkl LabelEncoder for drum classes
  models/feature_names.pkl    Ordered feature name list (ensures consistent
                               column alignment at inference time)

Run from the project root:
  python src/training/train_snd.py

Related modules:
  src/features/extraction_snd.py   — extracts the same features at inference time
  src/prediction/predict_snd.py    — loads the artifacts saved here
"""

import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "final_drum_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Features / labels
X = df.drop(columns=["label"])
y = df["label"]

# Save feature names
feature_names = X.columns.tolist()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="mlogloss"
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save everything
pickle.dump(model, open(MODEL_DIR / "drum_model.pkl", "wb"))
pickle.dump(scaler, open(MODEL_DIR / "scaler_snd.pkl", "wb"))
pickle.dump(le, open(MODEL_DIR / "label_encoder_snd.pkl", "wb"))
pickle.dump(feature_names, open(MODEL_DIR / "feature_names.pkl", "wb"))

print("\nModel saved successfully.")