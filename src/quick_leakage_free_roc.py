"""Quick leakage-aware ROC baselines for Diagnosis and Peritonitis.
Run from project root: .\.venv\Scripts\python.exe src\quick_leakage_free_roc.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score
from xgboost import XGBClassifier

OUT = Path('outputs/figures'); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_excel('data/raw/tabular/app_data.xlsx', sheet_name='All cases')
EXCLUDE = ['Diagnosis','Diagnosis_Presumptive','Management','Severity','Peritonitis',
           'Length_of_Stay','US_Number','Perforation','Appendicular_Abscess','Abscess_Location']
results=[]
for target in ['Diagnosis','Peritonitis']:
    d = df.loc[df[target].notna()].copy()
    yenc = LabelEncoder(); y = yenc.fit_transform(d[target].astype(str))
    X = d.drop(columns=[c for c in EXCLUDE if c in d.columns]).copy()
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X if c not in num]
    for c in cat: X[c] = X[c].astype('object').where(X[c].notna(), np.nan)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ('num',SimpleImputer(strategy='median',keep_empty_features=True),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='missing')),
                         ('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])
    model = XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,
                          eval_metric='mlogloss' if len(yenc.classes_)>2 else 'logloss',
                          random_state=42,n_jobs=2)
    pipe = Pipeline([('prep',prep),('model',model)])
    pipe.fit(xtr,ytr)
    p=pipe.predict_proba(xte); pred=pipe.predict(xte)
    plt.figure(figsize=(7,5))
    if len(yenc.classes_) == 2:
        score=roc_auc_score(yte,p[:,1]); fpr,tpr,_=roc_curve(yte,p[:,1]);
        plt.plot(fpr,tpr,label=f'XGBoost AUC = {score:.3f}',lw=2)
    else:
        # Macro one-vs-rest AUROC (unweighted average of per-class AUCs).
        scores=[]
        for i,name in enumerate(yenc.classes_):
            truth=(yte==i).astype(int)
            fpr,tpr,_=roc_curve(truth,p[:,i]); sc=auc(fpr,tpr);scores.append(sc)
            plt.plot(fpr,tpr,label=f'{name} AUC = {sc:.3f}',lw=2)
        score=float(np.mean(scores))
        plt.title(f'{target}: One-vs-Rest ROC (macro AUC = {score:.3f})')
    if len(yenc.classes_)==2: plt.title(f'{target}: leakage-aware holdout ROC')
    plt.plot([0,1],[0,1],'k--',alpha=.5,label='Chance')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');plt.grid(alpha=.2);plt.tight_layout()
    path=OUT/f'roc_{target.lower()}_leakage_free.png';plt.savefig(path,dpi=220);plt.close()
    row={'target':target,'auc':score,'accuracy':accuracy_score(yte,pred),
         'macro_f1':f1_score(yte,pred,average='macro'),'n_test':len(yte),
         'classes':' | '.join(yenc.classes_)}
    results.append(row)
    print(f'{target}: AUC={score:.4f} Accuracy={row["accuracy"]:.4f} MacroF1={row["macro_f1"]:.4f}')
    print('Classes:',list(yenc.classes_),'ROC:',path)
pd.DataFrame(results).to_csv('outputs/quick_leakage_free_roc_results.csv',index=False)
print('Saved outputs/quick_leakage_free_roc_results.csv')
print('NOTE: Exploratory holdout only. Confirm patient grouping, feature timing, and external validation before publication.')
