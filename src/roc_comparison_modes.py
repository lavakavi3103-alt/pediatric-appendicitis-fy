
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score

# Project paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Actual prediction files
OUR_CSV = OUTPUT / "peritonitis_final_probs.csv"
BASE_CSV = OUTPUT / "peritonitis_base_probs.csv"

# Published paper reported ROC-AUC
PUBLISHED_AUC = 0.967


def load_macro_roc(csv_path):
    """Calculate multiclass One-vs-Rest macro ROC from real predictions."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = ["y_true", "p_0", "p_1", "p_2"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{csv_path.name}: Missing columns {missing}"
        )

    y = df["y_true"].to_numpy(dtype=int)
    probabilities = df[["p_0", "p_1", "p_2"]].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability values")

    if not np.allclose(probabilities.sum(axis=1), 1, atol=0.02):
        raise ValueError("Class probabilities must sum to 1")

    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("All three classes must appear in test labels")

    fpr_grid = np.linspace(0, 1, 1001)
    tpr_values = []

    for class_id in range(3):
        binary_y = (y == class_id).astype(int)

        fpr, tpr, _ = roc_curve(
            binary_y,
            probabilities[:, class_id]
        )

        interpolated_tpr = np.interp(fpr_grid, fpr, tpr)
        interpolated_tpr[0] = 0.0

        tpr_values.append(interpolated_tpr)

    macro_tpr = np.mean(tpr_values, axis=0)
    macro_tpr[-1] = 1.0

    macro_auc = auc(fpr_grid, macro_tpr)

    ovr_auc = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=[0, 1, 2]
    )

    return fpr_grid, macro_tpr, macro_auc, ovr_auc


def main():

    # Our real DSA + XGBoost predictions
    our_fpr, our_tpr, our_auc, our_ovr_auc = load_macro_roc(
        OUR_CSV
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Published model: only plot if actual prediction CSV exists
    if BASE_CSV.exists():

        base_fpr, base_tpr, base_auc, base_ovr_auc = load_macro_roc(
            BASE_CSV
        )

        ax.plot(
            base_fpr,
            base_tpr,
            color="#2563eb",
            linewidth=2.5,
            label=f"XGBoost + ADASYN (AUC = {base_auc:.4f})"
        )

        print(f"Published model CSV macro curve AUC: {base_auc:.4f}")
        print(f"Published model OvR macro AUC: {base_ovr_auc:.4f}")

    else:
        print(
            "Published model prediction CSV unavailable. "
            "Blue curve cannot be calculated."
        )
        print(f"Published paper reported AUC: {PUBLISHED_AUC:.3f}")

    # Our model: actual predictions
    ax.plot(
        our_fpr,
        our_tpr,
        color="#ea580c",
        linewidth=2.5,
        label=f"DSA + XGBoost (AUC = {our_auc:.4f})"
    )

    # Chance reference
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    ax.set_title(
        "Peritonitis ROC Comparison",
        fontsize=14,
        fontweight="bold"
    )

    # No axis text labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)

    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.grid(alpha=0.2)

    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    fig.tight_layout()

    png_path = FIGURES / "roc_peritonitis_comparison.png"
    pdf_path = FIGURES / "roc_peritonitis_comparison.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Our model macro curve AUC: {our_auc:.4f}")
    print(f"Our model OvR macro AUC: {our_ovr_auc:.4f}")
    print("Saved:", png_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
