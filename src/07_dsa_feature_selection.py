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

# ============================================================
# 1. LOAD DATA (same as 05_pipeline.py)
# ============================================================
df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
TARGET = "Diagnosis"
df = df[df[TARGET].notna()].copy()

drop_cols = ["Diagnosis_Presumptive", "US_Number", "Management", "Severity", "Peritonitis"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=[TARGET] + drop_cols)
y = LabelEncoder().fit_transform(df[TARGET])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str).replace("nan", np.nan)
    X[c] = LabelEncoder().fit_transform(X[c].fillna("missing"))
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

feature_names = X.columns.tolist()
n_features = len(feature_names)
print("Total features:", n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_np = X_train.values
X_test_np = X_test.values

# ============================================================
# 2. FITNESS FUNCTION: selected features vechi accuracy check
# ============================================================
def fitness(mask):
    mask = mask > 0.5
    if mask.sum() == 0:
        return 1.0
    model = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    model.fit(X_train_np[:, mask], y_train)
    pred = model.predict(X_test_np[:, mask])
    acc = accuracy_score(y_test, pred)
    penalty = 0.001 * (mask.sum() / n_features)
    return 1 - acc + penalty  # minimize pannanum

# ============================================================
# 3. CUSTOM DUCK SWARM ALGORITHM (DSA)
#    Based on: Zhu, Y. et al. (2021), "Duck Swarm Algorithm"
#    Binary version-ku sigmoid transform use pannuvom
# ============================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

rng = np.random.default_rng(42)
POP_SIZE = 15
EPOCHS = 20

# Population-a random-a start pannuvom [0,1] range-la
pop = rng.random((POP_SIZE, n_features))
fitness_vals = np.array([fitness(ind) for ind in pop])

best_idx = np.argmin(fitness_vals)
best_pos = pop[best_idx].copy()
best_fit = fitness_vals[best_idx]

print("\nRunning custom Duck Swarm Algorithm...")
print(f"Initial best fitness: {best_fit:.4f}")

for epoch in range(EPOCHS):
    for i in range(POP_SIZE):
        r1, r2, r3, r4 = rng.random(4)
        mean_pos = pop.mean(axis=0)

        if r1 < 0.5:
            # Searching phase: best duck-a nokki move, cosine oscillation-oda
            new_pos = best_pos + r2 * (best_pos - pop[i]) * np.cos(2 * np.pi * r3)
        else:
            # Foraging phase: best-um mean-um nokki move pannum
            new_pos = pop[i] + r2 * (best_pos - pop[i]) + r4 * (mean_pos - pop[i])

        # Bounds-ku uள்ளே vechukanum [0,1]
        new_pos = np.clip(new_pos, 0, 1)

        new_fit = fitness(sigmoid(new_pos - 0.5))
        old_fit = fitness_vals[i]

        if new_fit < old_fit:
            pop[i] = new_pos
            fitness_vals[i] = new_fit

    gen_best_idx = np.argmin(fitness_vals)
    if fitness_vals[gen_best_idx] < best_fit:
        best_fit = fitness_vals[gen_best_idx]
        best_pos = pop[gen_best_idx].copy()

    print(f"Epoch {epoch+1}/{EPOCHS} | Best fitness: {best_fit:.4f}")

# ============================================================
# 4. FINAL SELECTED FEATURES
# ============================================================
best_mask = sigmoid(best_pos - 0.5) > 0.5
dsa_features = [f for f, m in zip(feature_names, best_mask) if m]
print(f"\nDSA selected {len(dsa_features)} features:", dsa_features)

final_model = XGBClassifier(eval_metric="logloss", random_state=42)
final_model.fit(X_train[dsa_features], y_train)
pred = final_model.predict(X_test[dsa_features])
proba = final_model.predict_proba(X_test[dsa_features])[:, 1]

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
print(f"\n[DSA-selected features] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

result = pd.DataFrame([{
    "name": "DSA-selected features",
    "accuracy": acc, "f1": f1, "auc": auc, "n_features": len(dsa_features)
}])
result.to_csv(OUT / "phase2_dsa_results.csv", index=False)
print("\nSaved: outputs/phase2_dsa_results.csv")

with open(OUT / "dsa_selected_features.txt", "w") as f:
    f.write("\n".join(dsa_features))
print("Saved: outputs/dsa_selected_features.txt")