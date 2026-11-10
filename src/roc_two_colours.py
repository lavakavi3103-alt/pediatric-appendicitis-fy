"""Plot two genuine diagnosis ROC curves, with different colours and no axis titles.
Run from project root:
  .\.venv\Scripts\python.exe src\roc_two_colours.py
Requires real outputs/presumptive_all_probs.csv and presumptive_dsa_probs.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ('XGBoost - All Features', ROOT / 'outputs/presumptive_all_probs.csv', '#1565C0'),
    ('DSA + XGBoost', ROOT / 'outputs/presumptive_dsa_probs.csv', '#D84315'),
]

def load_predictions(path):
    df = pd.read_csv(path)
    needed = {'y_true', 'p_0', 'p_1'}
    if not needed.issubset(df.columns):
        raise ValueError(f'{path.name}: required columns are y_true, p_0, p_1')
    y = df['y_true'].to_numpy(dtype=int)
    p = df['p_1'].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError(f'{path.name}: expected binary labels 0 and 1')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f'{path.name}: invalid probabilities')
    return y, p

def main():
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    reference_y = None
    for name, path, colour in MODELS:
        if not path.is_file():
            raise FileNotFoundError(f'Missing prediction CSV: {path}')
        y, p = load_predictions(path)
        if reference_y is None:
            reference_y = y
        elif len(y) != len(reference_y) or not np.array_equal(y, reference_y):
            raise ValueError('CSV labels differ: use identical test patients in identical order for comparison.')
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colour, linewidth=2.2, label=f'{name} (AUC = {auc:.3f})')
        print(f'{name}: AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], linestyle=':', color='#888888', linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    ax.set_title('ROC Curve Comparison', fontsize=12)
    fig.tight_layout()
    out = ROOT / 'figures/roc_two_colours'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(out) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}.png and {out}.pdf')

if __name__ == '__main__':
    main()
