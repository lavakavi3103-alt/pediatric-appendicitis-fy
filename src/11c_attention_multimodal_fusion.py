"""Exploratory gated-attention clinical + precomputed ultrasound CNN embeddings.
Run: .\.venv\Scripts\python.exe src\11c_attention_multimodal_fusion.py
WARNING: CNN embedding provenance is not verified; NOT a leakage-free image evaluation.
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(min(4,torch.get_num_threads()))
DATA=Path('data/processed/fusion_dataset.csv')
OUT=Path('outputs'); FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA); df=df[df['Diagnosis'].notna()].reset_index(drop=True)
img_cols=[c for c in df if c.startswith('img_feat_')]
if len(img_cols)!=512: print('WARNING: expected 512 image embeddings, found',len(img_cols))
if not img_cols: raise ValueError('No img_feat_ columns')
excluded={'Diagnosis','Diagnosis_Presumptive','Management','Severity','Peritonitis','Length_of_Stay','Perforation','Appendicular_Abscess','Abscess_Location','US_Number','US_Number_int','has_images'}
tab_cols=[c for c in df if c not in excluded and c not in img_cols and not c.lower().startswith('unnamed') and not any(s in c.lower() for s in ['patient_id','image_path','filename','diagnosis','target','label'])]
y=(df['Diagnosis'].astype(str).str.strip().str.lower()=='no appendicitis').astype(int).to_numpy()
if len(np.unique(y))!=2: raise ValueError('Need two diagnosis classes')
if 'US_Number' in df:
    ids=df['US_Number'].astype('string')
    groups=np.array([str(v) if pd.notna(v) and str(v).strip() else f'missing_{i}' for i,v in enumerate(ids)])
    trval,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df,y,groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+1).split(trval,y[trval],groups[trval]))
    tr,va=trval[tr],trval[va]
    assert not (set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    print('US_Number group overlaps: 0 (US_Number must still be verified as patient ID)')
else:
    trval,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=SEED,stratify=y)
    tr,va=train_test_split(trval,test_size=.2,random_state=SEED+1,stratify=y[trval])
    print('WARNING: row-level split; no patient grouping')
for ix in (tr,va,te):
    if len(np.unique(y[ix]))!=2: raise ValueError('A split has only one class; adjust seed')
num=[c for c in tab_cols if pd.api.types.is_numeric_dtype(df[c])]
cat=[c for c in tab_cols if c not in num]
transformers=[]
if num: transformers.append(('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num))
if cat: transformers.append(('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
pre=ColumnTransformer(transformers,sparse_threshold=0)
Ttr=pre.fit_transform(df.iloc[tr][tab_cols]); Tva=pre.transform(df.iloc[va][tab_cols]); Tte=pre.transform(df.iloc[te][tab_cols])
imp=SimpleImputer(strategy='median',keep_empty_features=True)
sc=StandardScaler()
img=df[img_cols].apply(pd.to_numeric,errors='coerce')
Itr=sc.fit_transform(imp.fit_transform(img.iloc[tr])); Iva=sc.transform(imp.transform(img.iloc[va])); Ite=sc.transform(imp.transform(img.iloc[te]))
print('Rows train/validation/test:',len(tr),len(va),len(te),'| Clinical:',Ttr.shape[1],'| CNN:',Itr.shape[1])
print('IMPORTANT: Existing CNN embeddings may have seen held-out patients during CNN training.')
class Model(nn.Module):
    def __init__(self,td,im,mode):
        super().__init__(); self.mode=mode
        self.t=nn.Sequential(nn.Linear(td,64),nn.ReLU(),nn.Dropout(.25))
        self.i=nn.Sequential(nn.Linear(im,64),nn.ReLU(),nn.Dropout(.25))
        self.gate=nn.Linear(128,2)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,2))
    def forward(self,t,i):
        a=self.t(t) if self.mode!='image' else torch.zeros((len(t),64),device=t.device)
        b=self.i(i) if self.mode!='clinical' else torch.zeros((len(i),64),device=i.device)
        if self.mode=='fusion':
            weights=torch.softmax(self.gate(torch.cat([a,b],dim=1)),dim=1)
            h=weights[:,0:1]*a+weights[:,1:2]*b
        else: h=a if self.mode=='clinical' else b
        return self.head(h)

def fit(mode):
    m=Model(Ttr.shape[1],Itr.shape[1],mode)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss()
    to=lambda a:torch.tensor(np.asarray(a,dtype=np.float32))
    a,b=to(Ttr),to(Itr); c,d=to(Tva),to(Iva); e,f=to(Tte),to(Ite)
    labels=torch.tensor(y[tr],dtype=torch.long)
    best=float('inf'); patience=0; best_state=None
    for epoch in range(100):
        m.train()
        for ix in torch.randperm(len(a)).split(64):
            opt.zero_grad(); loss=loss_fn(m(a[ix],b[ix]),labels[ix]);loss.backward();opt.step()
        m.eval()
        with torch.no_grad(): val_loss=loss_fn(m(c,d),torch.tensor(y[va],dtype=torch.long)).item()
        if val_loss < best-0.0001:
            best=val_loss; patience=0;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            patience+=1
            if patience>=12: break
    m.load_state_dict(best_state);m.eval()
    with torch.no_grad(): p=torch.softmax(m(e,f),dim=1)[:,1].numpy()
    pred=(p>=.5).astype(int)
    metrics={'model':mode,'accuracy':accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro'),'auc':roc_auc_score(y[te],p),'test_n':len(te)}
    print(mode,metrics,'epochs:',epoch+1)
    print(classification_report(y[te],pred,target_names=['appendicitis','no appendicitis'],zero_division=0))
    return metrics,p,m

results=[]; curves=[]
for mode in ['clinical','image','fusion']:
    met,p,model=fit(mode);results.append(met);curves.append((mode,p))
    if mode=='fusion':torch.save({'state_dict':model.state_dict(),'tabular_columns':tab_cols,'image_columns':img_cols,'clinical_input_dim':Ttr.shape[1],'image_input_dim':Itr.shape[1],'warning':'Pretrained CNN embedding provenance unverified'},OUT/'attention_fusion_exploratory.pth')
pd.DataFrame(results).to_csv(OUT/'attention_fusion_exploratory_results.csv',index=False)
plt.figure(figsize=(7,6))
for name,p in curves:
    fpr,tpr,_=roc_curve(y[te],p);auc=roc_auc_score(y[te],p)
    plt.plot(fpr,tpr,label=f'{name} AUC={auc:.3f}')
plt.plot([0,1],[0,1],'k--',alpha=.5)
plt.xlabel('False positive rate (no appendicitis)');plt.ylabel('True positive rate');plt.title('Exploratory attention fusion — Diagnosis');plt.legend();plt.tight_layout()
plt.savefig(FIG/'roc_attention_fusion_exploratory.png',dpi=180);plt.close()
print('Saved:',OUT/'attention_fusion_exploratory_results.csv',FIG/'roc_attention_fusion_exploratory.png')
print('CAUTION: NOT leakage-free end-to-end until CNN embeddings are regenerated using train-only CNN and independent test patients.')
