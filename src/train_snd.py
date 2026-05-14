import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("data/processed/final_drum_dataset.csv")

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
pickle.dump(model, open("models/drum_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler_snd.pkl", "wb"))
pickle.dump(le, open("models/label_encoder_snd.pkl", "wb"))
pickle.dump(feature_names, open("models/feature_names.pkl", "wb"))

print("\nModel saved successfully.")