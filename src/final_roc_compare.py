"""Two ROC comparison figures. Published curves are approximate tracings of Fig. 6,
NOT original predictions. Do not represent them as exact original ROC data.
Requires: numpy, pandas, matplotlib, scikit-learn.
Run from project root: .\.venv\Scripts\python.exe src\final_roc_compare.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

# Approximate visually traced coordinates from the user-provided published Fig. 6.
# These are illustrations of the reported curves, NOT original raw ROC points.
PUBLISHED_DIAGNOSIS = np.array([
    [0,0], [.001,.90], [.040,.94], [.075,.956], [.110,.969],
    [.150,.982], [.235,.995], [1,1]
], dtype=float)
# Peritonitis Fig. 6(b) shows THREE class-specific curves, not one original macro curve.
# Approximate class traces are interpolated to produce an illustrative macro curve.
PUB_PERI_CLASSES = [
    np.array([[0,0],[.003,.94],[.03,.98],[.07,1],[1,1]]),
    np.array([[0,0],[.008,.40],[.02,.56],[.04,.72],[.08,.86],[.15,.95],[.25,.98],[.6,.995],[1,1]]),
    np.array([[0,0],[.008,.18],[.02,.50],[.04,.69],[.08,.82],[.15,.90],[.30,.94],[.5,.97],[.8,.99],[1,1]])
]

def read_predictions(filename, multiclass=False):
    path = ROOT / 'outputs' / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing prediction CSV: {path}')
    df = pd.read_csv(path)
    cols = sorted((c for c in df if c.startswith('p_')), key=lambda c:int(c[2:]))
    if 'y_true' not in df or len(cols) < 2:
        raise ValueError(f'{filename}: expected y_true,p_0,p_1,...')
    y = df['y_true'].astype(int).to_numpy()
    p = df[cols].to_numpy(dtype=float)
    if not multiclass and p.shape[1] != 2:
        raise ValueError(f'{filename}: expected 2 probability columns')
    if multiclass and p.shape[1] != 3:
        raise ValueError(f'{filename}: expected 3 probability columns')
    return y, p

def multiclass_macro_curve(y, p):
    grid = np.linspace(0,1,1001)
    curves = []
    for c in range(p.shape[1]):
        yy = (y == c).astype(int)
        if len(np.unique(yy)) != 2:
            raise ValueError(f'Class {c} missing from test labels; macro ROC unavailable')
        fpr, tpr, _ = roc_curve(yy, p[:,c])
        curves.append(np.interp(grid, fpr, tpr))
    macro = np.mean(curves, axis=0)
    return grid, macro, np.trapezoid(macro, grid)

def style(ax):
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    ax.grid(color='#e2e2e2', linewidth=.55)
    # No False Positive Rate / True Positive Rate labels, per request.
    ax.set_xlabel('')
    ax.set_ylabel('')

def save(fig, name):
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: figures/{name}.png and figures/{name}.pdf')

def diagnosis():
    y, p = read_predictions('presumptive_dsa_probs.csv')
    fpr, tpr, _ = roc_curve(y, p[:,1], pos_label=1)
    auc = roc_auc_score(y,p[:,1])
    fig, ax = plt.subplots(figsize=(6.2,5.1))
    ax.plot(PUBLISHED_DIAGNOSIS[:,0], PUBLISHED_DIAGNOSIS[:,1],
            color='#2563eb', linewidth=2.1,
            label='Soft Voting + Borderline-SMOTE (reported AUC 0.9916) *')
    ax.plot(fpr,tpr,color='#e76f00',linewidth=2.1,
            label=f'DSA + XGBoost (test AUC {auc:.4f})')
    style(ax)
    ax.set_title('Diagnosis ROC comparison',fontsize=12)
    ax.legend(loc='lower right',fontsize=8,framealpha=.96)
    fig.text(.5,.01,'* Published curve approximately traced from Fig. 6(a); not original ROC coordinates.',
             ha='center',fontsize=7)
    save(fig,'roc_diagnosis_final')
    print(f'Diagnosis actual DSA AUC: {auc:.4f}')

def peritonitis():
    y, p = read_predictions('peritonitis_final_probs.csv', multiclass=True)
    grid, macro, auc = multiclass_macro_curve(y,p)
    pub_grid = np.linspace(0,1,1001)
    pub_macro = np.mean([np.interp(pub_grid,pts[:,0],pts[:,1]) for pts in PUB_PERI_CLASSES],axis=0)
    fig, ax = plt.subplots(figsize=(6.2,5.1))
    ax.plot(pub_grid,pub_macro,color='#2563eb',linewidth=2.1,
            label='XGBoost + ADASYN (reported AUC 0.967) *')
    ax.plot(grid,macro,color='#e76f00',linewidth=2.1,
            label=f'DSA + XGBoost (CSV macro ROC AUC {auc:.4f}) **')
    style(ax)
    ax.set_title('Peritonitis ROC comparison',fontsize=12)
    ax.legend(loc='lower right',fontsize=8,framealpha=.96)
    fig.text(.5,.012,'* Approximate macro curve from published class-wise Fig. 6(b), not original predictions.',
             ha='center',fontsize=7)
    fig.text(.5,-.007,'** CSV model identity and class order must be verified before publication.',
             ha='center',fontsize=7)
    save(fig,'roc_peritonitis_final')
    print(f'Peritonitis CSV macro curve AUC: {auc:.4f} (earlier reported 0.7225 may differ)')

if __name__ == '__main__':
    diagnosis()
    peritonitis()
    print('NOTE: Published blue curves are approximate figure tracings, NOT exact ROC data.')
