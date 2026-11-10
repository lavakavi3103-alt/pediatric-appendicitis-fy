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

df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
print("Loaded:", df.shape)

TARGET = "Diagnosis"
df = df[df[TARGET].notna()].copy()
print("After dropping missing target:", df.shape)

drop_cols = ["Diagnosis_Presumptive", "US_Number", "Management", "Severity", "Peritonitis"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=[TARGET] + drop_cols)
y = LabelEncoder().fit_transform(df[TARGET])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"\nCategorical columns: {len(cat_cols)} | Numeric columns: {len(num_cols)}")

for c in cat_cols:
    X[c] = X[c].astype(str).replace("nan", np.nan)
    X[c] = LabelEncoder().fit_transform(X[c].fillna("missing"))

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

print("Missing values remaining:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    print(f"\n[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc, "n_features": X_tr.shape[1]}

results = []

print("\n" + "=" * 60)
print("BASELINE: all features, XGBoost, no balancing")
print("=" * 60)
base_model = XGBClassifier(eval_metric="logloss", random_state=42)
results.append(evaluate(base_model, X_train, y_train, X_test, y_test, "Baseline (all features)"))

print("\n" + "=" * 60)
print("BORUTA feature selection")
print("=" * 60)
try:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators="auto", random_state=42, max_iter=50)
    boruta_selector.fit(X_train.values, y_train)

    boruta_features = X_train.columns[boruta_selector.support_].tolist()
    print(f"Boruta selected {len(boruta_features)} features:", boruta_features)

    if len(boruta_features) > 0:
        model_b = XGBClassifier(eval_metric="logloss", random_state=42)
        results.append(evaluate(
            model_b, X_train[boruta_features], y_train, X_test[boruta_features], y_test,
            "Boruta-selected features"
        ))
    else:
        print("Boruta selected 0 features, skipping.")
except Exception as e:
    print("Boruta failed:", e)

print("\n" + "=" * 60)
print("SMOTE balancing (on all features)")
print("=" * 60)
try:
    from imblearn.over_sampling import SMOTE

    print("Before SMOTE:", np.bincount(y_train))
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print("After SMOTE:", np.bincount(y_train_sm))

    model_sm = XGBClassifier(eval_metric="logloss", random_state=42)
    results.append(evaluate(model_sm, X_train_sm, y_train_sm, X_test, y_test, "Baseline + SMOTE"))
except Exception as e:
    print("SMOTE failed:", e)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(OUT / "phase2_results.csv", index=False)
print("\nSaved: outputs/phase2_results.csv")