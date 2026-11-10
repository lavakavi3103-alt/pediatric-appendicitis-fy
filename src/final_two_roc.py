import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PUBLISHED = {
    'diagnosis': ('Soft Voting + Borderline-SMOTE', 0.992),
    'peritonitis': ('XGBoost + ADASYN', 0.967),
}


def read_predictions(path):
    df = pd.read_csv(path)
    cols = sorted((c for c in df.columns if c.startswith('p_') and c[2:].isdigit()), key=lambda c: int(c[2:]))
    if 'y_true' not in df or len(cols) < 2:
        raise ValueError(f'{path}: expected y_true,p_0,p_1,...')
    y = df.y_true.to_numpy(dtype=int)
    P = df[cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P > 1):
        raise ValueError(f'{path}: invalid probabilities')
    if not np.allclose(P.sum(axis=1), 1, atol=0.02):
        raise ValueError(f'{path}: probabilities do not sum to 1')
    return y, P


def macro_ovr(y, P):
    grid = np.linspace(0, 1, 1001)
    curves = []
    for c in range(P.shape[1]):
        yy = (y == c).astype(int)
        if len(np.unique(yy)) != 2:
            raise ValueError(f'Class {c} missing positives or negatives in test set')
        fpr, tpr, _ = roc_curve(yy, P[:, c])
        curves.append(np.interp(grid, fpr, tpr))
    return grid, np.mean(curves, axis=0)


def read_published_curve(path):
    df = pd.read_csv(path)
    if not {'fpr','tpr'}.issubset(df.columns):
        raise ValueError(f'{path}: expected fpr,tpr columns')
    fpr = df.fpr.to_numpy(dtype=float)
    tpr = df.tpr.to_numpy(dtype=float)
    if not (np.all(np.isfinite(fpr)) and np.all(np.isfinite(tpr)) and np.all((fpr >= 0)&(fpr <= 1)) and np.all((tpr >= 0)&(tpr <= 1)) and np.all(np.diff(fpr) >= 0)):
        raise ValueError(f'{path}: invalid ROC coordinates')
    return fpr, tpr


def plot_task(task, prediction_path, output_path, published_curve_path=None):
    y, P = read_predictions(prediction_path)
    if task == 'diagnosis':
        if P.shape[1] != 2:
            raise ValueError('Diagnosis requires two probability columns')
        fpr, tpr, _ = roc_curve(y, P[:, 1], pos_label=1)
        auc = roc_auc_score(y, P[:, 1])
        our_name = 'DSA + XGBoost'
        auc_method = 'binary'
    else:
        if P.shape[1] != 3:
            raise ValueError('Peritonitis requires three probability columns')
        fpr, tpr = macro_ovr(y, P)
        auc = float(np.trapezoid(tpr, fpr))
        our_name = 'DSA + XGBoost (macro OvR)'
        auc_method = 'macro OvR (interpolated curve)'

    pub_name, pub_auc = PUBLISHED[task]
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(fpr, tpr, color='#E07A20', lw=2.2, label=f'{our_name} (AUC {auc:.4f})')
    if published_curve_path:
        pfpr, ptpr = read_published_curve(published_curve_path)
        ax.plot(pfpr, ptpr, color='#2468B4', lw=2.2, label=f'{pub_name} (reported AUC {pub_auc:.3f})')
        ax.text(0.03, 0.05, 'Published curve digitized/obtained from paper; verify source and class definition',
                fontsize=6.5, transform=ax.transAxes, color='0.35')
    else:
        ax.text(0.03, 0.07, f'Published: {pub_name}\nReported AUC = {pub_auc:.3f}\n(ROC coordinates unavailable; curve not plotted)',
                fontsize=8, transform=ax.transAxes, va='bottom',
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))
    ax.plot([0,1],[0,1], ls=':', color='0.65', lw=1, label='Chance')
    ax.set_xlim(0,1); ax.set_ylim(0,1.01)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_title('Diagnosis ROC' if task == 'diagnosis' else 'Peritonitis ROC (macro one-vs-rest)')
    ax.grid(color='0.9', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.95)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=300)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print(f'{task}: proposed model AUC={auc:.4f} ({auc_method}); published reported AUC={pub_auc:.3f}')
    print(f'Saved: {out.with_suffix(".png")} and {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two separate ROC figures from real predictions; published curves optional')
    parser.add_argument('--diagnosis', default='outputs/presumptive_dsa_probs.csv')
    parser.add_argument('--peritonitis', default='outputs/peritonitis_final_probs.csv')
    parser.add_argument('--published-diagnosis-curve', help='Optional real/digitized CSV: fpr,tpr')
    parser.add_argument('--published-peritonitis-curve', help='Optional real/digitized CSV: fpr,tpr')
    parser.add_argument('--outdir', default='figures')
    args = parser.parse_args()
    plot_task('diagnosis', args.diagnosis, str(Path(args.outdir)/'roc_diagnosis_comparison'), args.published_diagnosis_curve)
    plot_task('peritonitis', args.peritonitis, str(Path(args.outdir)/'roc_peritonitis_comparison'), args.published_peritonitis_curve)
