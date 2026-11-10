"""Leakage-conscious CatBoost + Optuna peritonitis experiment.
Run from project root: python src\10d_peritonitis_catboost_optuna.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data/processed/clinical_radiomics_fusion.csv'
FEATURES = OUT / 'dsa_selected_features.txt'
if not DATA.exists() or not FEATURES.exists():
    raise FileNotFoundError('Missing clinical_radiomics_fusion.csv or dsa_selected_features.txt')

df = pd.read_csv(DATA)
if 'Peritonitis' not in df:
    raise ValueError('Peritonitis target not found')
clinical = [s.strip() for s in FEATURES.read_text(encoding='utf-8').splitlines() if s.strip()]
# Exclude known post-outcome and complication proxies conservatively.
blocked = {'Length_of_Stay', 'Management', 'Diagnosis_Presumptive', 'Diagnosis',
           'Severity', 'Peritonitis', 'Perforation', 'Appendicular_Abscess',
           'Abscess_Location'}
clinical = [c for c in clinical if c in df.columns and c not in blocked]
# Radiomics generated from ultrasound ROI; select actual feature columns, not identifiers.
rad_prefixes = ('glcm_', 'lbp_')
rad_names = {'mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'median', 'p10', 'p90', 'iqr', 'entropy'}
radiomics = [c for c in df.columns if c not in blocked and (c.startswith(rad_prefixes) or c in rad_names)]
cols = list(dict.fromkeys(clinical + radiomics))
if not cols:
    raise ValueError('No features found')
d = df.loc[df['Peritonitis'].notna()].copy()
# Keep class labels in report for clear mapping.
y = d['Peritonitis'].astype(str)
X = d[cols].copy()
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    else:
        X[c] = X[c].fillna('__missing__').astype(str)
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
# Stratified 60/20/20 train/validation/test. Test is untouched until tuning ends.
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)
# Training-only imputation for numeric columns.
num_cols = [c for c in cols if c not in cat_cols]
medians = X_train[num_cols].median().fillna(0)
for part in (X_train, X_val, X_test):
    part.loc[:, num_cols] = part[num_cols].fillna(medians)
print(f'Patients: {len(d)} | Clinical: {len(clinical)} | Radiomics: {len(radiomics)}')
print(f'Train/validation/test: {len(X_train)}/{len(X_val)}/{len(X_test)}')
print('Classes:', y.value_counts().to_dict())

# Fixed test split; tune ONLY on validation. Optimize macro F1 for minority-class fairness.
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 250, 850, step=150),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 12, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 3.0),
        'loss_function': 'MultiClass', 'verbose': False, 'thread_count': -1,
        'random_seed': 42,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_cols)
    pred = np.asarray(model.predict(X_val)).ravel().astype(str)
    trial.set_user_attr('val_accuracy', float(accuracy_score(y_val, pred)))
    return f1_score(y_val, pred, average='macro', zero_division=0)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
print('Best validation macro F1:', round(study.best_value, 4))
print('Best validation accuracy:', round(study.best_trial.user_attrs['val_accuracy'], 4))
print('Best parameters:', study.best_params)

# Refit on train+validation; evaluate on untouched test once.
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
model = CatBoostClassifier(**study.best_params, loss_function='MultiClass', verbose=False,
                           thread_count=-1, random_seed=42)
model.fit(X_final, y_final, cat_features=cat_cols)
pred = np.asarray(model.predict(X_test)).ravel().astype(str)
proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro', zero_division=0)
try:
    auc = roc_auc_score(y_test, proba, labels=model.classes_, multi_class='ovr')
except ValueError:
    auc = float('nan')
print('\n=== UNTOUCHED TEST RESULTS ===')
print(f'Accuracy: {acc:.4f} ({acc*100:.2f}%) | Macro F1: {f1:.4f} | AUC: {auc:.4f}')
print(classification_report(y_test, pred, zero_division=0))
print('Confusion matrix (labels order):', list(model.classes_))
print(confusion_matrix(y_test, pred, labels=model.classes_))
path = OUT / 'peritonitis_catboost_optuna_results.csv'
pd.DataFrame([{'Model':'CatBoost + Optuna', 'N':len(d), 'Train':len(X_train),
               'Validation':len(X_val), 'Test':len(X_test), 'Accuracy':acc,
               'MacroF1':f1, 'AUC':auc, 'NTrials':20}]).to_csv(path, index=False)
model.save_model(str(OUT / 'peritonitis_catboost_optuna.cbm'))
print('\nSaved:', path)
print('NOTE: DSA features may have been selected on full data; this is not fully nested validation.')
print('NOTE: Do not directly compare this holdout result with prior 5-fold CV 76.05%.')
