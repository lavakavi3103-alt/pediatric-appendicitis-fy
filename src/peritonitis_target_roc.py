
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. AUC VALUES
# ==========================================
BASE_AUC = 0.967
TARGET_AUC = 0.987

# These are illustrative ROC curves.
# TARGET_AUC is not an achieved experimental result.

fpr = np.linspace(0, 1, 1000)

def create_target_curve(auc_value):
    exponent = auc_value / (1.0 - auc_value)
    return 1.0 - (1.0 - fpr) ** exponent

base_tpr = create_target_curve(BASE_AUC)
target_tpr = create_target_curve(TARGET_AUC)

# ==========================================
# 3. CREATE ROC GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Base paper model
ax.plot(
    fpr,
    base_tpr,
    color="#2563eb",
    linewidth=2.5,
    label=f"XGBoost + ADASYN (Reported AUC = {BASE_AUC:.3f})"
)

# Our project's desired target
ax.plot(
    fpr,
    target_tpr,
    color="#ea580c",
    linewidth=2.5,
    label=f"DSA + XGBoost (Target AUC = {TARGET_AUC:.3f})"
)

# Random classifier reference line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    color="gray",
    linewidth=1.2
)

# ==========================================
# 4. GRAPH FORMATTING
# ==========================================
ax.set_title(
    "Peritonitis ROC-AUC Comparison",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)

ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))

# Axis titles omitted as requested
ax.set_xlabel("")
ax.set_ylabel("")

ax.grid(alpha=0.2)

ax.legend(
    loc="lower right",
    fontsize=9,
    frameon=True
)

# Small neutral note, not a red warning footer
fig.text(
    0.5,
    0.015,
    "Illustrative target scenario; curves are not measured ROC results.",
    ha="center",
    fontsize=8,
    color="#666666"
)

fig.tight_layout(rect=[0, 0.045, 1, 1])

# ==========================================
# 5. SAVE GRAPH
# ==========================================
png_path = OUT / "peritonitis_target_roc.png"
pdf_path = OUT / "peritonitis_target_roc.pdf"

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

print("=" * 45)
print("Peritonitis ROC-AUC Comparison")
print("=" * 45)
print(f"Published AUC: {BASE_AUC:.3f}")
print(f"Target AUC:    {TARGET_AUC:.3f}")
print(f"Target improvement: {(TARGET_AUC - BASE_AUC) * 100:.1f} percentage points")
print("NOTE: Curves are illustrative, not experimental.")
print("Saved:", png_path)
print("Saved:", pdf_path)
