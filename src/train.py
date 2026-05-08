import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "music_5sec_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Assume you have:
# - 'label' column (genre)
# - 'track_id' column (IMPORTANT)

X = df.drop(columns=["label", "track_id"])
y = df["label"]
groups = df["track_id"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split by song_id (group split manually)
unique_songs = df["track_id"].unique()

train_songs, test_songs = train_test_split(
    unique_songs, test_size=0.2, random_state=42
)

train_idx = df["track_id"].isin(train_songs)
test_idx = df["track_id"].isin(test_songs)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03, 
    subsample=0.8, 
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) 

predictions = model.predict(X_test)

# 4. Check the Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Save everything
pickle.dump(model, open(MODEL_DIR / "model.pkl", "wb"))
pickle.dump(scaler, open(MODEL_DIR / "scaler.pkl", "wb"))
pickle.dump(le, open(MODEL_DIR / "label_encoder.pkl", "wb"))