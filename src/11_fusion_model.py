import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/fusion_dataset.csv")
print("Fusion dataset shape:", df.shape)

TARGET = "Diagnosis"
df = df[df[TARGET].notna()].copy()
y = LabelEncoder().fit_transform(df[TARGET])

img_feat_cols = [c for c in df.columns if c.startswith("img_feat_")]
print(f"Image feature columns: {len(img_feat_cols)}")

with open("outputs/dsa_selected_features.txt") as f:
    dsa_features = [line.strip() for line in f if line.strip()]
dsa_features = [c for c in dsa_features if c in df.columns]
print(f"DSA tabular features available: {len(dsa_features)}")

X_tab = df[dsa_features].copy()
for c in X_tab.columns:
    if not pd.api.types.is_numeric_dtype(X_tab[c]):
        X_tab[c] = X_tab[c].astype(str).replace("nan", np.nan)
        X_tab[c] = LabelEncoder().fit_transform(X_tab[c].fillna("missing"))
    else:
        X_tab[c] = X_tab[c].fillna(X_tab[c].median())

X_img = df[img_feat_cols].copy()
X_fusion = pd.concat([X_tab.reset_index(drop=True), X_img.reset_index(drop=True)], axis=1)

X_train_t, X_test_t, X_train_i, X_test_i, X_train_f, X_test_f, y_train, y_test = train_test_split(
    X_tab, X_img, X_fusion, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(X_tr, y_tr, X_te, y_te, name):
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc, "n_features": X_tr.shape[1]}

print("\n" + "=" * 60)
print("ABLATION STUDY: Tabular-only vs Image-only vs Fusion")
print("=" * 60)

results = []
results.append(evaluate(X_train_t, y_train, X_test_t, y_test, "Tabular-only (DSA features)"))
results.append(evaluate(X_train_i, y_train, X_test_i, y_test, "Image-only (CNN features)"))
results.append(evaluate(X_train_f, y_train, X_test_f, y_test, "FUSION (Tabular + Image)"))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(OUT / "phase4_fusion_results.csv", index=False)
print("\nSaved: outputs/phase4_fusion_results.csv")