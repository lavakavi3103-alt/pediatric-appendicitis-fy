import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data/processed/clinical_radiomics_fusion.csv'
FEATURES = OUT / 'dsa_selected_features.txt'

if not DATA.exists() or not FEATURES.exists():
    raise FileNotFoundError('Missing fusion CSV or dsa_selected_features.txt')

df = pd.read_csv(DATA)
selected = [s.strip() for s in FEATURES.read_text(encoding='utf-8').splitlines() if s.strip()]
# Post-outcome and potentially post-diagnosis features excluded conservatively.
excluded = {
    'Length_of_Stay', 'Management', 'Diagnosis_Presumptive',
    'Diagnosis', 'Severity', 'Peritonitis',
    'Perforation', 'Appendicular_Abscess', 'Abscess_Location',
    'US_Number', 'US_Number_int', 'patient_id'
}
clinical = list(dict.fromkeys(c for c in selected if c in df.columns and c not in excluded))
rad_names = {'mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'median', 'p10', 'p90', 'entropy'}
radiomics = [c for c in df.columns if (c in rad_names or c.startswith(('glcm_', 'lbp_', 'rad_'))) and c not in excluded]
features = list(dict.fromkeys(clinical + radiomics))
if not features:
    raise ValueError('No input features found')
print(f'Clinical features: {len(clinical)} | Radiomics features: {len(radiomics)}')

# Split conformal with finite-sample corrected quantile.
def conformal_quantile(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))

rows = []
for target in ['Diagnosis', 'Severity', 'Peritonitis']:
    d = df[df[target].notna()].copy()
    if d.empty:
        continue
    X = d[features].copy()
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(d[target].astype(str))
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 5:
        print(f'Skipping {target}: insufficient samples per class')
        continue

    # 60% train / 20% calibration / 20% untouched test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    numeric = X_train.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(('num', SimpleImputer(strategy='median', add_indicator=False), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    n_classes = len(y_encoder.classes_)
    model = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        random_state=42, n_jobs=-1)
    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)

    cal_proba = pipe.predict_proba(X_cal)
    test_proba = pipe.predict_proba(X_test)
    scores = 1.0 - cal_proba[np.arange(len(y_cal)), y_cal]
    qhat = conformal_quantile(scores, alpha=0.10)
    prediction_sets = (1.0 - test_proba) <= qhat
    # Empty sets are allowed by this conformal method.
    set_sizes = prediction_sets.sum(axis=1)
    covered = prediction_sets[np.arange(len(y_test)), y_test]
    pred = test_proba.argmax(axis=1)
    singleton_mask = set_sizes == 1
    singleton_acc = (float(np.mean(pred[singleton_mask] == y_test[singleton_mask]))
                     if singleton_mask.any() else np.nan)

    row = {
        'Target': target, 'N': len(d), 'Train': len(y_train),
        'Calibration': len(y_cal), 'Test': len(y_test),
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'MacroF1': round(f1_score(y_test, pred, average='macro'), 4),
        'TargetCoverage': 0.90,
        'EmpiricalCoverage': round(float(covered.mean()), 4),
        'MeanSetSize': round(float(set_sizes.mean()), 4),
        'SingletonCount': int(singleton_mask.sum()),
        'SingletonAccuracy': round(singleton_acc, 4) if np.isfinite(singleton_acc) else np.nan
    }
    rows.append(row)
    print(f'\n=== {target} ===')
    print('Classes:', list(y_encoder.classes_))
    print('Train/Calibration/Test:', len(y_train), len(y_cal), len(y_test))
    print('Accuracy:', f"{row['Accuracy']:.2%}")
    print('Macro F1:', row['MacroF1'])
    print('Target coverage: 90% | Actual coverage:', f"{row['EmpiricalCoverage']:.2%}")
    print('Mean prediction set size:', row['MeanSetSize'])
    print('Singleton predictions:', row['SingletonCount'], '| Singleton accuracy:', row['SingletonAccuracy'])

if not rows:
    raise RuntimeError('No target could be evaluated')
path = OUT / 'uncertainty_results.csv'
pd.DataFrame(rows).to_csv(path, index=False)
print('\nSaved:', path)
print('NOTE: Earlier DSA selection may have full-dataset bias; not fully nested validation.')
