import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DATA = ROOT / 'data' / 'processed' / 'clinical_radiomics_fusion.csv'
df = pd.read_csv(DATA)
selected = [x.strip() for x in (OUT / 'dsa_selected_features.txt').read_text(encoding='utf-8').splitlines() if x.strip()]

# Exclude outcome, post-admission, and potentially outcome-revealing variables.
excluded = {
    'Length_of_Stay', 'Management', 'Diagnosis_Presumptive',
    'Diagnosis', 'Severity', 'Peritonitis', 'Perforation',
    'Appendicular_Abscess', 'Abscess_Location'
}
clinical = [c for c in selected if c in df.columns and c not in excluded]

# Radiomics fields produced by roi_features.py (GLCM, LBP, intensity stats).
rad_prefixes = ('glcm_', 'lbp_', 'rad_')
rad_names = {'mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'median',
             'p10', 'p90', 'iqr', 'entropy', 'image_count'}
radiomics = [c for c in df.columns if (c.startswith(rad_prefixes) or c in rad_names) and c not in excluded]
radiomics = [c for c in radiomics if pd.api.types.is_numeric_dtype(df[c])]
print('Clinical features:', len(clinical), '| Radiomics features:', len(radiomics))
print('Excluded from selected clinical features:', [c for c in selected if c in excluded])
if not clinical or not radiomics:
    raise ValueError('Missing clinical or radiomics columns. Check the merged CSV.')

rows = []
for target in ('Diagnosis', 'Severity', 'Peritonitis'):
    if target not in df:
        print('Skipping missing target:', target)
        continue
    d = df[df[target].notna()].copy()
    y = LabelEncoder().fit_transform(d[target].astype(str))
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 5:
        print('Skipping target with too few examples per class:', target, counts)
        continue
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for config, columns in [('Clinical only', clinical),
                            ('Clinical + Radiomics', list(dict.fromkeys(clinical + radiomics)))]:
        X = d[columns].copy()
        nums = X.select_dtypes(include=np.number).columns.tolist()
        cats = [c for c in X.columns if c not in nums]
        transformers = []
        if nums:
            transformers.append(('numeric', SimpleImputer(strategy='median', keep_empty_features=True), nums))
        if cats:
            transformers.append(('categorical', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore'))
            ]), cats))
        prep = ColumnTransformer(transformers)
        model = XGBClassifier(random_state=42, n_jobs=2,
                              eval_metric='mlogloss' if len(counts) > 2 else 'logloss')
        pipe = Pipeline([('preprocessing', prep), ('model', model)])
        proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba', n_jobs=1)
        pred = proba.argmax(axis=1)
        f1 = f1_score(y, pred, average='macro' if len(counts) > 2 else 'binary', zero_division=0)
        auc = roc_auc_score(y, proba, multi_class='ovr') if len(counts) > 2 else roc_auc_score(y, proba[:, 1])
        row = {'Target': target, 'Config': config, 'N': len(d),
               'Accuracy': round(accuracy_score(y, pred), 4),
               'F1': round(f1, 4), 'AUC': round(auc, 4)}
        rows.append(row)
        print(row, flush=True)
OUT.mkdir(exist_ok=True)
result = pd.DataFrame(rows)
path = OUT / 'radiomics_fusion_corrected_results.csv'
result.to_csv(path, index=False)
print('\nSaved:', path)
print('NOTE: DSA selection may still have full-dataset selection bias; not a fully nested validation.')
