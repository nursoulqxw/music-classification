"""
Generates a system architecture diagram PNG for the Music Genre Classifier.
Output: reports/architecture_diagram.png

Run from the project root:
    python src/analysis/generate_diagram.py
"""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = ROOT_DIR / "reports" / "architecture_diagram.png"

# ── Palette ──────────────────────────────────────────────────────────
C = {
    "user":     "#4C72B0",
    "backend":  "#6c757d",
    "feature":  "#55A868",
    "model":    "#C44E52",
    "ensemble": "#8172B2",
    "result":   "#CCB974",
    "section":  "#f0f4f8",
    "border":   "#dee2e6",
    "text":     "#212529",
    "arrow":    "#495057",
}


def box(ax, cx, cy, w, h, label, sub=None, color="#4C72B0", fs=10):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white",
        linewidth=1.8, zorder=4, alpha=0.93,
    )
    ax.add_patch(rect)
    if sub:
        ax.text(cx, cy + h * 0.15, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=5)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=fs - 2.5, color="white", alpha=0.88, zorder=5,
                style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=5)


def section(ax, x, y, w, h, title, color="#f0f4f8"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=C["border"],
        linewidth=1.2, zorder=1, alpha=0.6,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.18, title,
            ha="center", va="top", fontsize=8.5,
            color=C["backend"], fontweight="bold", zorder=2,
            style="italic")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=C["arrow"], lw=1.6,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=3,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=7.5,
                color=C["arrow"], va="center", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(9, 9.6, "Music Genre Classifier — System Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C["text"])
    ax.text(9, 9.2, "End-to-end pipeline: audio upload → feature extraction → ensemble inference → genre result",
            ha="center", va="center", fontsize=9, color=C["backend"])

    # ── Section backgrounds ───────────────────────────────────────────
    section(ax, 0.3, 0.4, 3.1, 8.4, "USER / FRONTEND", "#eef6fb")
    section(ax, 3.7, 0.4, 5.0, 8.4, "FEATURE ENGINEERING", "#eefbf0")
    section(ax, 9.0, 0.4, 5.5, 8.4, "ML ENSEMBLE", "#fdf3f3")
    section(ax, 14.8, 0.4, 2.9, 8.4, "OUTPUT", "#fdfaf0")

    # ── USER / FRONTEND ───────────────────────────────────────────────
    box(ax, 1.85, 7.8, 2.5, 0.9, "Web Browser",
        "Upload audio file", C["user"])
    arrow(ax, 1.85, 7.35, 1.85, 6.55)

    box(ax, 1.85, 6.1, 2.5, 0.9, "FastAPI Backend",
        "POST /predict  :6767", C["backend"])
    arrow(ax, 1.85, 5.65, 1.85, 4.85)

    box(ax, 1.85, 4.4, 2.5, 0.9, "Temp file",
        ".mp3 / .wav / .flac / .ogg", C["backend"])
    arrow(ax, 3.1, 4.4, 3.7, 4.4)          # → feature engineering

    # ── FEATURE ENGINEERING ───────────────────────────────────────────
    box(ax, 6.2, 7.8, 4.2, 0.9, "librosa — Load & Segment",
        "Split into 5-second windows", C["feature"])
    arrow(ax, 6.2, 7.35, 6.2, 6.55)

    box(ax, 6.2, 6.1, 4.2, 0.9, "Feature Extraction (×N segments)",
        "13 MFCC · 12 Chroma · Centroid · Bandwidth\nRolloff · Flatness · ZCR · RMS · Tempo",
        C["feature"], fs=8.5)
    arrow(ax, 6.2, 5.65, 6.2, 4.85)

    box(ax, 6.2, 4.4, 4.2, 0.9, "StandardScaler  +  PCA",
        "Normalize · Reduce for SVM path", C["feature"])

    # connect from temp file → librosa
    arrow(ax, 3.7, 4.4, 4.1, 4.4)

    # connect feature engineering → model section
    arrow(ax, 8.3, 4.4, 9.0, 4.4)

    ax.text(5.2, 4.0, "63 features\n× N segments", ha="center",
            fontsize=7.5, color=C["backend"], style="italic")

    # ── ML ENSEMBLE ───────────────────────────────────────────────────
    model_x = [10.0, 11.5, 12.5, 13.7]
    labels   = ["XGBoost\n(40%)", "SVM + PCA\n(25%)", "ExtraTrees\n(20%)", "CatBoost\n(15%)"]
    for mx, lbl in zip(model_x, labels):
        box(ax, mx, 6.8, 1.2, 1.3, lbl, color=C["model"], fs=8)
        arrow(ax, mx, 6.15, mx, 5.55)

    # fan-in arrows to ensemble
    for mx in model_x:
        arrow(ax, mx, 5.25, 11.8, 4.75)

    box(ax, 11.8, 4.4, 4.2, 0.9, "Weighted Probability Average",
        "P_ensemble = 0.40·P_xgb + 0.25·P_svm + 0.20·P_et + 0.15·P_cb",
        C["ensemble"], fs=8.5)
    arrow(ax, 11.8, 3.95, 11.8, 3.15)

    box(ax, 11.8, 2.7, 4.2, 0.9, "Segment-level Average",
        "Mean probabilities across all 5-sec windows", C["ensemble"])

    # input arrow from feature section to models (top)
    for mx in model_x:
        arrow(ax, 9.0, 4.4, mx - 0.0, 6.15)

    # also connect input to model row
    arrow(ax, 9.0, 4.4, 9.8, 6.8)

    # ── OUTPUT ────────────────────────────────────────────────────────
    arrow(ax, 13.9, 2.7, 14.8, 2.7)

    box(ax, 16.25, 4.5, 2.5, 1.0, "Genre Prediction",
        "e.g. HipHop  81%", C["result"])
    box(ax, 16.25, 2.7, 2.5, 1.0, "Confidence Scores",
        "All 10 genres ranked", C["result"])
    box(ax, 16.25, 0.95, 2.5, 1.0, "JSON Response",
        "→ Web UI display", C["user"])

    arrow(ax, 16.25, 4.0, 16.25, 3.2)
    arrow(ax, 16.25, 2.2, 16.25, 1.45)

    # ── CNN optional path ─────────────────────────────────────────────
    box(ax, 6.2, 2.0, 4.2, 0.9, "CNN Path (optional)",
        "ResNet18 on mel spectrograms", "#6c757d")
    ax.annotate(
        "", xy=(14.8, 2.7), xytext=(8.3, 2.0),
        arrowprops=dict(arrowstyle="->", color="#6c757d",
                        lw=1.2, linestyle="dashed",
                        connectionstyle="arc3,rad=-0.2"),
        zorder=3,
    )
    ax.text(11.5, 1.5, "if model_type=cnn", fontsize=7.5,
            color="#6c757d", ha="center", style="italic")

    # ── Legend ────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C["user"],     label="Frontend / API"),
        mpatches.Patch(color=C["feature"],  label="Feature Engineering"),
        mpatches.Patch(color=C["model"],    label="ML Models"),
        mpatches.Patch(color=C["ensemble"], label="Ensemble / Aggregation"),
        mpatches.Patch(color=C["result"],   label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.01, 0.01), ncol=5,
              fontsize=8, framealpha=0.8, edgecolor=C["border"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
