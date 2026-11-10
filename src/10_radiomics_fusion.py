import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

OUT = Path("outputs")
df = pd.read_csv("data/processed/clinical_radiomics_fusion.csv")

with open(OUT / "dsa_selected_features.txt") as f:
    dsa_features = [line.strip() for line in f if line.strip()]
dsa_features = [c for c in dsa_features if c in df.columns]

radiomics_cols = [c for c in df.columns if c in (
    "mean", "std", "skew", "kurtosis", "min", "max", "median", "p10", "p90", "entropy"
) or c.startswith("glcm_")]

def prep(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).replace("nan", np.nan)
            X[c] = LabelEncoder().fit_transform(X[c].fillna("missing"))
        else:
            X[c] = X[c].fillna(X[c].median())
    return X

targets = ["Diagnosis", "Severity", "Peritonitis"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for t in targets:
    if t not in df.columns:
        print("Skipping (column not found):", t)
        continue
    d = df[df[t].notna()].copy()
    y = LabelEncoder().fit_transform(d[t].astype(str))
    if len(np.unique(y)) < 2:
        print("Skipping (only one class):", t)
        continue
    multi = len(np.unique(y)) > 2

    for name, cols in [("Clinical only", dsa_features),
                       ("Clinical + Radiomics", dsa_features + radiomics_cols)]:
        cols = [c for c in cols if c != t]
        X = prep(d[cols])
        model = XGBClassifier(eval_metric="mlogloss" if multi else "logloss", random_state=42)
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro" if multi else "binary")
        auc = roc_auc_score(y, proba, multi_class="ovr") if multi else roc_auc_score(y, proba[:, 1])
        rows.append({"Target": t, "Config": name, "N": len(d),
                     "Accuracy": round(acc, 4), "F1": round(f1, 4), "AUC": round(auc, 4)})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT / "radiomics_fusion_results.csv", index=False)
print("\nSaved: outputs/radiomics_fusion_results.csv")