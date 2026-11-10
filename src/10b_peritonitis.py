import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

USE_RADIOMICS = True   # False pannaa clinical-only run aagum

OUT = Path("outputs")
df = pd.read_csv("data/processed/clinical_radiomics_fusion.csv")
df = df[df["Peritonitis"].notna()].copy()
y = LabelEncoder().fit_transform(df["Peritonitis"].astype(str))

drop = {"Diagnosis", "Severity", "Management", "Peritonitis", "Diagnosis_Presumptive",
        "US_Number", "US_Number_int", "patient_id"}
radiomics = {"mean", "std", "skew", "kurtosis", "min", "max", "median", "p10", "p90", "entropy"}
rad_cols = [c for c in df.columns if c in radiomics or c.startswith("glcm_")]
base_cols = [c for c in df.columns if c not in drop and c not in rad_cols]
cols = base_cols + (rad_cols if USE_RADIOMICS else [])

X = df[cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.factorize(X[c].astype(str))[0]
X = X.dropna(axis=1, how="all")
print("Samples:", len(X), "| Features:", X.shape[1], "| Class counts:", np.bincount(y))

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", random_state=42, n_jobs=-1)

def cat():
    return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                              verbose=0, random_state=42)

def gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.1, random_state=42)

def vote():
    return VotingClassifier([("xgb", xgb()), ("cat", cat()), ("gb", gb())], voting="soft")

models = {"XGBoost": xgb, "CatBoost": cat, "GradBoost": gb, "Soft Voting": vote}
samplers = {
    "No resampling": None,
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for s_name, sampler in samplers.items():
    for m_name, make in models.items():
        steps = [("imp", SimpleImputer(strategy="median"))]
        if sampler is not None:
            steps.append(("samp", sampler))
        steps.append(("clf", make()))
        pipe = Pipeline(steps)
        try:
            proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")
        except Exception as e:
            print("Skipped", s_name, m_name, "->", e)
            continue
        pred = proba.argmax(axis=1)
        rows.append({
            "Sampler": s_name, "Model": m_name,
            "Accuracy": round(accuracy_score(y, pred), 4),
            "MacroF1": round(f1_score(y, pred, average="macro"), 4),
            "AUC": round(roc_auc_score(y, proba, multi_class="ovr"), 4),
        })
        print(rows[-1])

res = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
print("\n=== Results (sorted by accuracy) ===")
print(res.to_string(index=False))
res.to_csv(OUT / "peritonitis_results.csv", index=False)
print("\nSaved: outputs/peritonitis_results.csv")