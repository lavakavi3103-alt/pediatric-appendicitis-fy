import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).parent))
from make_roc import dump_probabilities

SEED = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/clinical_radiomics_fusion.csv")

drop_common = {"Diagnosis", "Severity", "Management", "Peritonitis",
               "Diagnosis_Presumptive", "US_Number", "US_Number_int",
               "patient_id", "Length_of_Stay", "Perforation", "Appendicular_Abscess"}

def prep_xy(target):
    d = df[df[target].notna()].copy()
    y = LabelEncoder().fit_transform(d[target].astype(str))
    X = d.drop(columns=[c for c in drop_common if c in d.columns])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c].astype(str))[0]
        X[c] = X[c].fillna(X[c].median())
    return X, y

# ============================================================
# Diagnosis — all features vs DSA-selected features
# ============================================================
X, y = prep_xy("Diagnosis")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

model_all = XGBClassifier(eval_metric="logloss", random_state=SEED)
model_all.fit(Xtr, ytr)
dump_probabilities(model_all, Xte, yte, "outputs/presumptive_all_probs.csv")
print("Saved: outputs/presumptive_all_probs.csv")

with open(OUT / "dsa_selected_features.txt") as f:
    dsa_features = [line.strip() for line in f if line.strip()]
dsa_features = [c for c in dsa_features if c in Xtr.columns]

model_dsa = XGBClassifier(eval_metric="logloss", random_state=SEED)
model_dsa.fit(Xtr[dsa_features], ytr)
dump_probabilities(model_dsa, Xte[dsa_features], yte, "outputs/presumptive_dsa_probs.csv")
print("Saved: outputs/presumptive_dsa_probs.csv")

# ============================================================
# Peritonitis — final XGBoost model
# ============================================================
X, y = prep_xy("Peritonitis")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
n_classes = len(np.unique(y))

model_p = XGBClassifier(objective="multi:softprob", num_class=n_classes,
                        eval_metric="mlogloss", random_state=SEED)
model_p.fit(Xtr, ytr)
dump_probabilities(model_p, Xte, yte, "outputs/peritonitis_final_probs.csv")
print("Saved: outputs/peritonitis_final_probs.csv")

print("\nAll probability files generated successfully.")