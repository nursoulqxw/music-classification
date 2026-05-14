import pickle
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "music_5sec_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["label", "track_id"])
    y = df["label"]
    groups = df["track_id"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    unique_songs = groups.unique()

    train_songs, test_songs = train_test_split(
        unique_songs,
        test_size=0.2,
        random_state=42
    )

    train_idx = df["track_id"].isin(train_songs)
    test_idx = df["track_id"].isin(test_songs)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    test_track_ids = groups[test_idx]

    # --- Standard Scaling ---
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- PCA for SVM ---
    pca = PCA(n_components=0.95, random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # ==========================================================
    # XGBoost
    # ==========================================================
    print("Training XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    xgb_model.fit(X_train_scaled, y_train)

    # ==========================================================
    # SVM
    # ==========================================================
    print("Training SVM...")

    svm_model = SVC(
        kernel='rbf',
        C=10,
        probability=True,
        random_state=42
    )

    svm_model.fit(X_train_pca, y_train)

    # ==========================================================
    # Extra Trees
    # ==========================================================
    print("Training Extra Trees...")

    extra_trees_model = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    extra_trees_model.fit(X_train_scaled, y_train)

    # ==========================================================
    # CatBoost
    # ==========================================================
    print("Training CatBoost...")

    catboost_model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    catboost_model.fit(X_train_scaled, y_train)

    # ==========================================================
    # Probability Extraction
    # ==========================================================
    print("Extracting probabilities...")

    xgb_probs = xgb_model.predict_proba(X_test_scaled)
    svm_probs = svm_model.predict_proba(X_test_pca)
    extra_trees_probs = extra_trees_model.predict_proba(X_test_scaled)
    catboost_probs = catboost_model.predict_proba(X_test_scaled)



    # ==========================================================
    # Weighted Soft Voting
    # ==========================================================
    blended_probs = (
        (xgb_probs * 0.40)
        + (svm_probs * 0.25)
        + (extra_trees_probs * 0.20)
        + (catboost_probs * 0.15)
    )

    # ==========================================================
    # Track-Level Aggregation
    # ==========================================================
    results_df = pd.DataFrame(
        blended_probs,
        columns=le.classes_
    )

    results_df["track_id"] = test_track_ids.values
    results_df["true_label"] = le.inverse_transform(y_test)

    # Mean aggregation over segments
    track_predictions = results_df.groupby("track_id").mean(numeric_only=True)

    track_true_labels = (
        results_df.groupby("track_id")["true_label"].first()
    )

    final_preds = track_predictions.idxmax(axis=1)

    # ==========================================================
    # METRICS
    # ==========================================================
    accuracy = accuracy_score(track_true_labels, final_preds)

    precision = precision_score(
        track_true_labels,
        final_preds,
        average="weighted"
    )

    recall = recall_score(
        track_true_labels,
        final_preds,
        average="weighted"
    )

    f1 = f1_score(
        track_true_labels,
        final_preds,
        average="weighted"
    )

    # --- ROC-AUC & PR-AUC ---
    y_true_encoded = le.transform(track_true_labels)

    y_true_binarized = label_binarize(
        y_true_encoded,
        classes=range(len(le.classes_))
    )

    roc_auc = roc_auc_score(
        y_true_binarized,
        track_predictions.values,
        multi_class="ovr",
        average="weighted"
    )

    pr_auc = average_precision_score(
        y_true_binarized,
        track_predictions.values,
        average="weighted"
    )

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================
    print("\n" + "=" * 60)
    print("FINAL TRACK-LEVEL RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(classification_report(
        track_true_labels,
        final_preds
    ))

    # ==========================================================
    # SAVE MODELS
    # ==========================================================
    print("Saving models and preprocessors to disk...")

    with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    with open(MODEL_DIR / "svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    with open(MODEL_DIR / "extra_trees_model.pkl", "wb") as f:
        pickle.dump(extra_trees_model, f)

    with open(MODEL_DIR / "catboost_model.pkl", "wb") as f:
        pickle.dump(catboost_model, f)

    with open(MODEL_DIR / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"\nAll artifacts saved successfully in '{MODEL_DIR}'.")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
