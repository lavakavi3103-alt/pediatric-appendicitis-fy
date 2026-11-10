"""Two-model ROC comparison using REAL binary test predictions.

Each CSV must contain y_true,p_0,p_1 with class 1 = appendicitis.
For a valid head-to-head comparison, use the same held-out patients,
label encoding, and test order for both files.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def load(path):
    df = pd.read_csv(path)
    required = {'y_true', 'p_0', 'p_1'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: required columns are y_true,p_0,p_1')
    y = df['y_true'].to_numpy()
    p = df['p_1'].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError(f'{path}: y_true must contain both 0 and 1')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f'{path}: p_1 must be probabilities between 0 and 1')
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published', required=True, help='Real test probabilities CSV for published model')
    ap.add_argument('--proposed', required=True, help='Real test probabilities CSV for proposed model')
    ap.add_argument('--published-name', default='Soft Voting + Borderline-SMOTE')
    ap.add_argument('--proposed-name', default='DSA + XGBoost')
    ap.add_argument('--out', default='figures/roc_two_models')
    args = ap.parse_args()
    y1, p1 = load(args.published)
    y2, p2 = load(args.proposed)
    if len(y1) != len(y2) or not np.array_equal(y1, y2):
        raise ValueError('Different test labels/order. Use identical test patients and order for direct comparison.')
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for name, y, p, style in [(args.published_name, y1, p1, '-'), (args.proposed_name, y2, p2, '--')]:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color='black', linestyle=style, linewidth=1.8,
                label=f'{name} (AUC = {auc:.3f})')
        print(f'{name}: AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], color='0.7', linestyle=':', linewidth=0.8, label='_nolegend_')
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('ROC Curve Comparison – Appendicitis Diagnosis')
    ax.legend(loc='lower right', frameon=True, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + '.png', dpi=300)
    fig.savefig(str(out) + '.pdf')
    plt.close(fig)
    print(f'Saved: {out}.png and {out}.pdf')

if __name__ == '__main__':
    main()
