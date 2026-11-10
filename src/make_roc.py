"""
make_roc.py
------------
Black-and-white, IEEE-style ROC curves from REAL test-set predictions.

INPUT: one CSV per model, with columns
        y_true, p_0, p_1, ... p_{K-1}
       (y_true = integer class index; p_k = predicted probability of class k)
       Use dump_probabilities() below to create these files from your model.

Binary target (presumptive diagnosis): all models are overlaid on ONE axes
(e.g. all features vs Boruta vs DSA), distinguished by line style.
Multiclass target (peritonitis): one-vs-rest curve for each class plus the
macro-average, one panel per model.

AUC is reported with a 95% stratified-bootstrap confidence interval. With a
small test set the CI is wide, report it.

Usage:
  python make_roc.py --inputs "All features=outputs/presumptive_all_probs.csv" \
                              "Boruta=outputs/presumptive_boruta_probs.csv" \
                              "DSA=outputs/presumptive_dsa_probs.csv" \
                     --out figures/roc_presumptive --classes "No appendicitis,Appendicitis"

  python make_roc.py --inputs "XGBoost=outputs/peritonitis_final_probs.csv" \
                     --out figures/roc_peritonitis --classes "No,Local,Generalized"

  python make_roc.py --selftest      # synthetic data, checks the plumbing only
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
LINESTYLES = ["-", "--", ":", "-."]


def dump_probabilities(model, X_test, y_test, path):
    """Call this once per model on the TEST set to create the CSV this script reads."""
    proba = model.predict_proba(X_test)
    df = pd.DataFrame(proba, columns=[f"p_{k}" for k in range(proba.shape[1])])
    df.insert(0, "y_true", np.asarray(y_test))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _load(path):
    df = pd.read_csv(path)
    pcols = [c for c in df.columns if c.startswith("p_")]
    return df["y_true"].to_numpy().astype(int), df[pcols].to_numpy(), len(pcols)


def _auc_ci(y_bin, score, n_boot=1000, seed=0):
    """AUC and 95% stratified bootstrap CI for a binary problem."""
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y_bin, score)
    pos, neg = np.where(y_bin == 1)[0], np.where(y_bin == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y_bin[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, lo, hi


def _frame(ax):
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=(0, (1, 2)))  # chance line
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("False positive rate (1 \u2212 specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", lw=0.5)


def plot_binary(models, classes, out):
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    _frame(ax)
    for i, (name, path) in enumerate(models):
        y, P, k = _load(path)
        assert k == 2, f"{name}: expected 2 probability columns, found {k}"
        auc, lo, hi = _auc_ci(y, P[:, 1])
        fpr, tpr, _ = roc_curve(y, P[:, 1])
        ax.plot(fpr, tpr, color="black", lw=1.4, ls=LINESTYLES[i % 4],
                label=f"{name} (AUC {auc:.3f} [{lo:.3f}, {hi:.3f}])")
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False, borderpad=0.5)
    _save(fig, out)


def plot_multiclass(models, classes, out):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4), squeeze=False)
    for ax, (name, path) in zip(axes[0], models):
        y, P, k = _load(path)
        names = classes if classes and len(classes) == k else [f"Class {c}" for c in range(k)]
        _frame(ax)
        grid = np.linspace(0, 1, 201)
        tprs = []
        for c in range(k):
            yb = (y == c).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            auc, lo, hi = _auc_ci(yb, P[:, c])
            fpr, tpr, _ = roc_curve(yb, P[:, c])
            tprs.append(np.interp(grid, fpr, tpr))
            ax.plot(fpr, tpr, color="black", lw=1.2, ls=LINESTYLES[c % 4],
                    label=f"{names[c]} (AUC {auc:.3f} [{lo:.2f}, {hi:.2f}], n={int(yb.sum())})")
        macro = np.mean(tprs, axis=0)
        ax.plot(grid, macro, color="0.45", lw=2.0,
                label=f"Macro-average (AUC {np.trapezoid(macro, grid):.3f})")
        ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False, borderpad=0.4)
        if n > 1:
            ax.set_title(name, fontsize=8)
    _save(fig, out)


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")  # vector, preferred by IEEE
    plt.close(fig)
    print(f"saved {out}.png and {out}.pdf")


def run(inputs, classes, out):
    models = []
    for item in inputs:
        name, path = item.split("=", 1)
        models.append((name.strip(), path.strip()))
    _, _, k = _load(models[0][1])
    cls = [c.strip() for c in classes.split(",")] if classes else None
    (plot_binary if k == 2 else plot_multiclass)(models, cls, out)


def selftest():
    import tempfile
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    tmp = tempfile.mkdtemp()
    for k, w in ((2, [0.6, 0.4]), (3, [0.7, 0.22, 0.08])):
        X, y = make_classification(n_samples=700, n_features=12, n_informative=6, n_classes=k,
                                   weights=w, class_sep=1.0, random_state=k)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        m = GradientBoostingClassifier(n_estimators=60, random_state=0).fit(Xtr, ytr)
        dump_probabilities(m, Xte, yte, f"{tmp}/m{k}.csv")
    run([f"Model A={tmp}/m2.csv", f"Model B={tmp}/m2.csv"], None, f"{tmp}/bin")
    run([f"Model A={tmp}/m3.csv"], "No,Local,Generalized", f"{tmp}/multi")
    for f in ("bin.png", "bin.pdf", "multi.png", "multi.pdf"):
        assert os.path.getsize(f"{tmp}/{f}") > 5000, f
    print("SELFTEST OK ->", tmp, "(synthetic data: never use these images in the paper)")
    return tmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help='"Name=path.csv" ...')
    ap.add_argument("--classes", default=None, help='comma separated class names')
    ap.add_argument("--out", default="figures/roc")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not a.inputs:
            ap.error("--inputs is required")
        run(a.inputs, a.classes, a.out)