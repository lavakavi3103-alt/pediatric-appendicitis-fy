import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / 'data/processed/clinical_radiomics_fusion.csv')
with open(OUT / 'dsa_selected_features.txt', encoding='utf-8') as f:
    selected = [s.strip() for s in f if s.strip()]
excluded = {'Length_of_Stay','Management','Diagnosis_Presumptive','Diagnosis','Severity','Peritonitis','Perforation','Appendicular_Abscess','Abscess_Location'}
clinical = [c for c in selected if c in df.columns and c not in excluded]
rad = [c for c in df.columns if c.startswith(('glcm_', 'lbp_')) or c in {'mean','std','skew','kurtosis','min','max','median','p10','p90','entropy','iqr','rad_image_count'}]
features = list(dict.fromkeys(clinical + rad))
d = df.loc[df['Peritonitis'].notna()].copy()
X = d[features].replace([np.inf, -np.inf], np.nan)
encoder = LabelEncoder()
y = encoder.fit_transform(d['Peritonitis'].astype(str))
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42)
num = X.select_dtypes(include=np.number).columns.tolist()
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([('num',SimpleImputer(strategy='median', add_indicator=False),num),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)],verbose_feature_names_out=False)
Xtr = pre.fit_transform(X_train)
Xva = pre.transform(X_val)
Xte = pre.transform(X_test)
print(f'Patients: {len(d)} | Clinical: {len(clinical)} | Radiomics: {len(rad)}')
print(f'Train/validation/test: {len(y_train)}/{len(y_val)}/{len(y_test)}')
print('Classes:', dict(zip(encoder.classes_, np.bincount(y))))

def params(trial):
    return dict(n_estimators=trial.suggest_int('n_estimators',150,700,step=50),learning_rate=trial.suggest_float('learning_rate',0.015,0.2,log=True),num_leaves=trial.suggest_int('num_leaves',7,63),max_depth=trial.suggest_int('max_depth',2,8),min_child_samples=trial.suggest_int('min_child_samples',5,45),subsample=trial.suggest_float('subsample',0.65,1.0),subsample_freq=1,colsample_bytree=trial.suggest_float('colsample_bytree',0.65,1.0),reg_lambda=trial.suggest_float('reg_lambda',0.01,10,log=True),verbosity=-1,random_state=42,n_jobs=2)

def objective(trial):
    m=LGBMClassifier(**params(trial))
    m.fit(Xtr,y_train)
    return accuracy_score(y_val,m.predict(Xva))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=20)
print('Best validation accuracy:',round(study.best_value,4))
print('Best parameters:',study.best_params)
# Refit preprocessing on combined development data; holdout test remains untouched.
pre_final = ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)])
Xdev=pre_final.fit_transform(X_dev)
Xtest=pre_final.transform(X_test)
model=LGBMClassifier(**study.best_params,subsample_freq=1,verbosity=-1,random_state=42,n_jobs=2)
model.fit(Xdev,y_dev)
pred=model.predict(Xtest)
prob=model.predict_proba(Xtest)
acc=accuracy_score(y_test,pred)
f1=f1_score(y_test,pred,average='macro',zero_division=0)
auc=roc_auc_score(y_test,prob,multi_class='ovr',labels=np.arange(len(encoder.classes_)))
print('\n=== UNTOUCHED TEST RESULTS ===')
print(f'Accuracy: {acc:.4f} ({acc*100:.2f}%) | Macro F1: {f1:.4f} | AUC: {auc:.4f}')
print(classification_report(y_test,pred,target_names=encoder.classes_,zero_division=0))
print('Confusion matrix (order):',list(encoder.classes_))
print(confusion_matrix(y_test,pred))
path=OUT/'peritonitis_lightgbm_optuna_results.csv'
pd.DataFrame([{'Model':'LightGBM + Optuna','N':len(d),'TestN':len(y_test),'Accuracy':acc,'MacroF1':f1,'AUC':auc,'ValidationAccuracy':study.best_value,'Features':len(features)}]).to_csv(path,index=False)
print('\nSaved:',path)
print('NOTE: Prior DSA feature selection may have full-dataset bias. Do not compare this holdout directly with prior 5-fold CV.')
