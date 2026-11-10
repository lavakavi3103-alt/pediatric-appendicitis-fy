import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42
OUT = Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel('data/raw/tabular/app_data.xlsx', sheet_name='All cases')
df = df[df['Diagnosis'].notna()].copy()

exclude = {
    'Diagnosis', 'Diagnosis_Presumptive', 'Management', 'Severity',
    'Peritonitis', 'Length_of_Stay', 'US_Number', 'Perforation',
    'Appendicular_Abscess', 'Abscess_Location'
}
X = df.drop(columns=[c for c in exclude if c in df.columns]).copy()
X = X.replace({pd.NA: np.nan})
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(df['Diagnosis'].astype(str))
if len(y_encoder.classes_) != 2:
    raise ValueError(f'Expected 2 diagnosis classes, got {list(y_encoder.classes_)}')

# Outer test set stays untouched during all selection and tuning.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
# DSA uses only an inner validation subset of the outer training set.
X_inner, X_val, y_inner, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED
)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
for frame in (X_inner, X_val, X_train, X_test):
    for c in cat_cols:
        frame[c] = frame[c].astype('object').where(frame[c].notna(), np.nan)

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', keep_empty_features=True), num_cols),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
], remainder='drop', verbose_feature_names_out=False)

Z_inner = preprocessor.fit_transform(X_inner)
Z_val = preprocessor.transform(X_val)
feature_names = preprocessor.get_feature_names_out().tolist()
n_features = len(feature_names)
print('Rows:', len(df), '| Train:', len(X_train), '| Untouched test:', len(X_test))
print('Classes:', list(y_encoder.classes_))
print('Encoded features:', n_features)

rng = np.random.default_rng(SEED)
POP_SIZE = 15
EPOCHS = 20

# Cache repeated masks so DSA does not retrain the same subset needlessly.
cache = {}
def fitness(position):
    mask = np.asarray(position) > 0.5
    if not mask.any():
        return 1.0
    key = mask.tobytes()
    if key not in cache:
        model = XGBClassifier(n_estimators=100, eval_metric='logloss',
                              random_state=SEED, n_jobs=2, verbosity=0)
        model.fit(Z_inner[:, mask], y_inner)
        acc = accuracy_score(y_val, model.predict(Z_val[:, mask]))
        cache[key] = 1 - acc + 0.001 * mask.mean()
    return cache[key]

pop = rng.random((POP_SIZE, n_features))
values = np.array([fitness(p) for p in pop])
best_index = np.argmin(values)
best_pos = pop[best_index].copy()
best_fit = values[best_index]
print('\nDSA optimization (inner validation only):')
for epoch in range(EPOCHS):
    mean_pos = pop.mean(axis=0)
    for i in range(POP_SIZE):
        r1, r2, r3, r4 = rng.random(4)
        if r1 < 0.5:
            candidate = best_pos + r2 * (best_pos - pop[i]) * np.cos(2 * np.pi * r3)
        else:
            candidate = pop[i] + r2 * (best_pos - pop[i]) + r4 * (mean_pos - pop[i])
        candidate = np.clip(candidate, 0, 1)
        score = fitness(candidate)
        if score < values[i]:
            pop[i], values[i] = candidate, score
    j = np.argmin(values)
    if values[j] < best_fit:
        best_pos, best_fit = pop[j].copy(), values[j]
    print(f'Epoch {epoch + 1}/{EPOCHS} | best fitness {best_fit:.4f}')

selected_mask = best_pos > 0.5
selected_features = [name for name, keep in zip(feature_names, selected_mask) if keep]
if not selected_features:
    raise RuntimeError('DSA selected no features')

# Reuse the same preprocessor fitted on inner training data.
Z_train = preprocessor.transform(X_train)
Z_test = preprocessor.transform(X_test)

model = XGBClassifier(n_estimators=100, eval_metric='logloss',
                      random_state=SEED, n_jobs=2, verbosity=0)
model.fit(Z_train[:, selected_mask], y_train)
pred = model.predict(Z_test[:, selected_mask])
prob = model.predict_proba(Z_test[:, selected_mask])[:, 1]
metrics = {
    'model': 'DSA XGBoost (outer holdout)',
    'accuracy': accuracy_score(y_test, pred),
    'f1': f1_score(y_test, pred),
    'auc': roc_auc_score(y_test, prob),
    'n_features': len(selected_features),
    'n_test': len(y_test),
}
print('\nSelected encoded features:', len(selected_features))
print('Holdout Accuracy: {:.2%} | F1: {:.4f} | AUC: {:.4f}'.format(
    metrics['accuracy'], metrics['f1'], metrics['auc']))
pd.DataFrame([metrics]).to_csv(OUT / 'phase2_dsa_leakage_free_results.csv', index=False)
(OUT / 'dsa_leakage_free_selected_features.txt').write_text(
    '\n'.join(selected_features), encoding='utf-8')
print('Saved: outputs/phase2_dsa_leakage_free_results.csv')
print('Saved: outputs/dsa_leakage_free_selected_features.txt')
print('NOTE: This is a single holdout evaluation; external validation is still needed.')


# Save deployment artifact
import joblib

artifact = {
    "preprocessor": preprocessor,
    "model": model,
    "selected_mask": selected_mask,
    "input_columns": X.columns.tolist(),
    "classes": y_encoder.classes_.tolist(),
    "selected_features": selected_features
}

joblib.dump(artifact, OUT / "diagnosis_deployment.pkl")
print("Deployment model saved successfully!")
