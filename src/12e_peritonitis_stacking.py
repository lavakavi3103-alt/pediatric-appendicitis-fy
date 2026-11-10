import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

SEED = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/clinical_radiomics_fusion.csv")
df = df[df["Peritonitis"].notna()].copy()
y = LabelEncoder().fit_transform(df["Peritonitis"].astype(str))

drop = {"Diagnosis", "Severity", "Management", "Peritonitis", "Diagnosis_Presumptive",
        "US_Number", "US_Number_int", "patient_id", "Length_of_Stay",
        "Perforation", "Appendicular_Abscess"}
X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.factorize(X[c].astype(str))[0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train/test: {len(ytrain)}/{len(ytest)} | Features: {X.shape[1]}")

base_learners = [
    ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          objective="multi:softprob", num_class=3,
                          eval_metric="mlogloss", random_state=SEED)),
    ("cat", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                verbose=False, random_state=SEED)),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)),
]

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    n_jobs=-1,
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("smotetomek", SMOTETomek(random_state=SEED)),
    ("stack", stack),
])

print("Training stacking ensemble (SMOTETomek + XGB/CatBoost/RF -> LogisticRegression)...")
pipe.fit(Xtrain, ytrain)

pred = pipe.predict(Xtest)
prob = pipe.predict_proba(Xtest)
acc = accuracy_score(ytest, pred)
f1 = f1_score(ytest, pred, average="macro", zero_division=0)
auc = roc_auc_score(ytest, prob, multi_class="ovr", average="weighted")

print(f"\n[STACKING + SMOTETomek] Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | AUC: {auc:.4f}")

pd.DataFrame([{"model": "Stacking+SMOTETomek", "accuracy": acc, "f1_macro": f1, "auc": auc}]).to_csv(
    OUT / "peritonitis_stacking_results.csv", index=False)
print("Saved: outputs/peritonitis_stacking_results.csv")
