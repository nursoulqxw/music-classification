import pickle
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "music_5sec_features.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "model_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPORT_DIR / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import seaborn as sns


ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.40,
    "SVM": 0.25,
    "Extra Trees": 0.20,
    "CatBoost": 0.15,
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts():
    required_paths = {
        "xgb": MODEL_DIR / "xgb_model.pkl",
        "svm": MODEL_DIR / "svm_model.pkl",
        "extra_trees": MODEL_DIR / "extra_trees_model.pkl",
        "catboost": MODEL_DIR / "catboost_model.pkl",
        "pca": MODEL_DIR / "pca.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "label_encoder": MODEL_DIR / "label_encoder.pkl",
    }

    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required model artifacts:\n{missing_text}")

    artifacts = {name: load_pickle(path) for name, path in required_paths.items()}

    legacy_path = MODEL_DIR / "model.pkl"
    if legacy_path.exists():
        artifacts["legacy"] = load_pickle(legacy_path)

    return artifacts


def prepare_test_data():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["label", "track_id"])
    y = df["label"]
    groups = df["track_id"]

    unique_songs = groups.unique()
    _, test_songs = train_test_split(unique_songs, test_size=0.2, random_state=42)
    test_idx = df["track_id"].isin(test_songs)

    return X[test_idx], y[test_idx], groups[test_idx]


def aggregate_track_predictions(probabilities, class_names, y_true, track_ids):
    results_df = pd.DataFrame(probabilities, columns=class_names)
    results_df["track_id"] = track_ids.values
    results_df["true_label"] = y_true.values

    track_probabilities = results_df.groupby("track_id")[class_names].mean()
    track_true_labels = results_df.groupby("track_id")["true_label"].first()
    track_predictions = track_probabilities.idxmax(axis=1)

    return track_true_labels, track_predictions, track_probabilities


def get_model_probabilities(models, X_scaled, X_pca):
    probabilities = {
        "XGBoost": models["xgb"].predict_proba(X_scaled),
        "SVM": models["svm"].predict_proba(X_pca),
        "Extra Trees": models["extra_trees"].predict_proba(X_scaled),
        "CatBoost": models["catboost"].predict_proba(X_scaled),
    }

    if "legacy" in models:
        probabilities["Legacy model.pkl"] = models["legacy"].predict_proba(X_scaled)

    ensemble_probs = sum(
        probabilities[name] * weight for name, weight in ENSEMBLE_WEIGHTS.items()
    )
    probabilities["Weighted Ensemble"] = ensemble_probs

    return probabilities


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def calculate_auc_metrics(y_true, probabilities, class_names, label_encoder):
    y_true_encoded = label_encoder.transform(y_true)
    y_true_bin = label_binarize(y_true_encoded, classes=range(len(class_names)))

    return {
        "roc_auc_ovr_macro": roc_auc_score(
            y_true_bin,
            probabilities,
            average="macro",
            multi_class="ovr",
        ),
        "pr_auc_macro": average_precision_score(
            y_true_bin,
            probabilities,
            average="macro",
        ),
    }


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - Track-Level Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = f"confusion_matrix_{slugify(model_name)}.png"
    plt.savefig(REPORT_DIR / filename, dpi=200)
    plt.close()


def save_performance_panel(y_true, y_pred, probabilities, class_names, model_name, label_encoder):
    y_true_encoded = label_encoder.transform(y_true)
    y_true_bin = label_binarize(y_true_encoded, classes=range(len(class_names)))

    roc_auc = roc_auc_score(
        y_true_bin,
        probabilities,
        average="macro",
        multi_class="ovr",
    )
    pr_auc = average_precision_score(y_true_bin, probabilities, average="macro")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"{model_name} Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probabilities.ravel())
    axes[1].plot(fpr, tpr, label=f"Macro ROC-AUC = {roc_auc:.3f}", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set_title(f"{model_name} ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true_bin.ravel(),
        probabilities.ravel(),
    )
    axes[2].plot(
        recall_vals,
        precision_vals,
        label=f"Macro PR-AUC = {pr_auc:.3f}",
        linewidth=2,
    )
    axes[2].set_title(f"{model_name} Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"performance_panel_{slugify(model_name)}.png", dpi=200)
    plt.close()


def save_model_comparison(metrics_df):
    sorted_df = metrics_df.sort_values("f1_macro", ascending=False)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=sorted_df, x="model", y="f1_macro", hue="model", palette="viridis")
    plt.ylim(0, 1)
    plt.title("Model Comparison - Track-Level Macro F1")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.xticks(rotation=25, ha="right")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison_f1_macro.png", dpi=200)
    plt.close()

    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    plot_df = sorted_df.set_index("model")[metric_columns]
    plt.figure(figsize=(10, 6))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Model Metrics Summary")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_metrics_heatmap.png", dpi=200)
    plt.close()


def save_per_class_f1(reports):
    rows = []
    for model_name, report in reports.items():
        for label, values in report.items():
            if isinstance(values, dict) and "f1-score" in values:
                rows.append(
                    {
                        "model": model_name,
                        "class": label,
                        "f1_score": values["f1-score"],
                    }
                )

    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(REPORT_DIR / "per_class_f1_scores.csv", index=False)

    pivot = per_class_df.pivot(index="model", columns="class", values="f1_score")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1)
    plt.title("Per-Class Track-Level F1 Scores")
    plt.xlabel("Class")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "per_class_f1_heatmap.png", dpi=200)
    plt.close()


def save_feature_importance(models, feature_names):
    importance_sources = {
        "XGBoost": models["xgb"],
        "Extra Trees": models["extra_trees"],
        "CatBoost": models["catboost"],
    }

    for model_name, model in importance_sources.items():
        if not hasattr(model, "feature_importances_"):
            continue

        importances = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importances.to_csv(
            REPORT_DIR / f"feature_importance_{slugify(model_name)}.csv",
            index=False,
        )

        top_features = importances.head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x="importance", y="feature", color="#4C72B0")
        plt.title(f"{model_name} - Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / f"feature_importance_{slugify(model_name)}.png",
            dpi=200,
        )
        plt.close()


def slugify(value):
    return value.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def main():
    print("Loading dataset and model artifacts...")
    models = load_artifacts()
    X_test, y_test, test_track_ids = prepare_test_data()

    class_names = list(models["label_encoder"].classes_)
    X_test_scaled = models["scaler"].transform(X_test)
    X_test_pca = models["pca"].transform(X_test_scaled)

    print("Calculating predictions and metrics...")
    model_probabilities = get_model_probabilities(models, X_test_scaled, X_test_pca)

    metrics_rows = []
    reports = {}

    for model_name, probabilities in model_probabilities.items():
        track_true, track_pred, track_probabilities = aggregate_track_predictions(
            probabilities,
            class_names,
            y_test,
            test_track_ids,
        )

        metrics = calculate_metrics(track_true, track_pred)
        metrics.update(
            calculate_auc_metrics(
                track_true,
                track_probabilities.values,
                class_names,
                models["label_encoder"],
            )
        )
        metrics["model"] = model_name
        metrics_rows.append(metrics)

        report = classification_report(
            track_true,
            track_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        reports[model_name] = report

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(REPORT_DIR / f"classification_report_{slugify(model_name)}.csv")
        save_confusion_matrix(track_true, track_pred, class_names, model_name)
        save_performance_panel(
            track_true,
            track_pred,
            track_probabilities.values,
            class_names,
            model_name,
            models["label_encoder"],
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[
        [
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "roc_auc_ovr_macro",
            "pr_auc_macro",
        ]
    ].sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(REPORT_DIR / "model_metrics_summary.csv", index=False)

    save_model_comparison(metrics_df)
    save_per_class_f1(reports)
    save_feature_importance(models, X_test.columns)

    print("\nModel metrics summary:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nReports and plots saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
