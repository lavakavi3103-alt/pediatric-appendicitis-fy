import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / 'data/processed/clinical_radiomics_fusion.csv')
df = df[df['Peritonitis'].notna()].copy()
y_enc = LabelEncoder()
y = y_enc.fit_transform(df['Peritonitis'].astype(str))
with open(OUT / 'dsa_selected_features.txt', encoding='utf-8') as f:
    selected = [line.strip() for line in f if line.strip()]
unsafe = {'Length_of_Stay','Management','Diagnosis','Severity','Peritonitis',
          'Diagnosis_Presumptive','Perforation','Appendicular_Abscess',
          'US_Number','US_Number_int','patient_id'}
clinical = [c for c in selected if c in df.columns and c not in unsafe]
rad = [c for c in df.columns if c.startswith(('glcm_','lbp_')) or c in
       {'mean','std','skew','kurtosis','min','max','median','p10','p90','entropy','iqr','rad_image_count'}]
cols = list(dict.fromkeys(clinical + rad))
X = df[cols].copy()
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].replace(r'^\s*$', np.nan, regex=True)
num = X.select_dtypes(include='number').columns.tolist()
cat = [c for c in X.columns if c not in num]
prep = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', add_indicator=False), num),
    ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                      ('enc',OneHotEncoder(handle_unknown='ignore'))]), cat)
], remainder='drop')

def xgb():
    return XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,
                         subsample=.8,colsample_bytree=.8,eval_metric='mlogloss',
                         random_state=42,n_jobs=2)
def catboost():
    return CatBoostClassifier(iterations=300,depth=5,learning_rate=.05,
                              verbose=False,random_seed=42,thread_count=2)
def lightgbm():
    return LGBMClassifier(n_estimators=250,max_depth=4,num_leaves=15,
                          learning_rate=.05,verbosity=-1,random_state=42,n_jobs=2)

def voting():
    return VotingClassifier([('xgb',xgb()),('cat',catboost()),('lgb',lightgbm())],voting='soft')
def stacking():
    return StackingClassifier([('xgb',xgb()),('cat',catboost()),('lgb',lightgbm())],
                              final_estimator=CatBoostClassifier(iterations=120,depth=3,
                              verbose=False,random_seed=42,thread_count=2),
                              cv=3,n_jobs=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,
                                    stratify=y,random_state=42)
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
models = {'Soft Voting':voting, 'Stacking':stacking, 'XGBoost':xgb,
          'CatBoost':catboost, 'LightGBM':lightgbm}
rows=[]
print('Patients:',len(X),'| Features:',len(cols),'| Train/test:',len(X_train),len(X_test))
print('Classes:',dict(zip(y_enc.classes_,np.bincount(y))))
print('Clinical:',len(clinical),'| Radiomics:',len(rad))
print('Excluded:',sorted(set(selected)&unsafe))
print('\nSelecting model using TRAIN-only cross-validation...')
for name, make in models.items():
    pipe=Pipeline([('prep',prep),('model',make())])
    scores=cross_val_score(pipe,X_train,y_train,cv=cv,scoring='accuracy',n_jobs=1)
    rows.append({'Model':name,'Train_CV_Accuracy':round(float(scores.mean()),4),
                 'Train_CV_SD':round(float(scores.std()),4)})
    print(name,':',round(float(scores.mean()),4),'+/-',round(float(scores.std()),4))
res=pd.DataFrame(rows).sort_values('Train_CV_Accuracy',ascending=False)
best=res.iloc[0]['Model']
print('\nSelected from training CV:',best)
final=Pipeline([('prep',prep),('model',models[best]())])
final.fit(X_train,y_train)
pred=final.predict(X_test)
print('\n=== HOLDOUT TEST (ONE-TIME EVALUATION) ===')
print('Accuracy:',round(accuracy_score(y_test,pred),4))
print('Macro F1:',round(f1_score(y_test,pred,average='macro'),4))
print(classification_report(y_test,pred,target_names=y_enc.classes_,zero_division=0))
print('Confusion matrix (order:',list(y_enc.classes_),'):')
print(confusion_matrix(y_test,pred))
res['Selected']=res['Model'].eq(best)
res['Holdout_Accuracy']=np.where(res['Selected'],round(accuracy_score(y_test,pred),4),np.nan)
res['Holdout_MacroF1']=np.where(res['Selected'],round(f1_score(y_test,pred,average='macro'),4),np.nan)
path=OUT/'peritonitis_fair_comparison.csv'
res.to_csv(path,index=False)
print('Saved:',path)
print('CAUTION: DSA features were previously selected; possible full-dataset selection bias remains.')
