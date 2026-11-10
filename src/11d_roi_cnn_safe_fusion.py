import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path('.')
OUT=ROOT/'outputs'; (OUT/'figures').mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv('data/processed/image_manifest_roi.csv')
df=pd.read_csv('data/processed/fusion_dataset.csv')
assert manifest['patient_id'].notna().all() and df['US_Number'].notna().all(), 'Missing patient IDs'
manifest['pid']=manifest['patient_id'].astype(int)
df['pid']=df['US_Number'].astype(int)
assert manifest['pid'].nunique()==len(df)==df['pid'].nunique()==578, 'Unexpected patient count / duplicate IDs'
assert set(manifest.pid)==set(df.pid), 'Manifest and clinical IDs do not match'
assert set(df.Diagnosis.dropna().unique())=={'appendicitis','no appendicitis'}
labels=df.set_index('pid')['Diagnosis'].map({'no appendicitis':0,'appendicitis':1})
assert labels.notna().all()
assert (manifest.groupby('pid')['diagnosis'].first().str.lower().map({'no appendicitis':0,'appendicitis':1}).sort_index()==labels.sort_index()).all(), 'Image/clinical labels differ'
# EXACT original 09h split: do not sort or stratify!
original_ids=manifest['patient_id'].unique()
cnn_train_ids,cnn_test_ids=train_test_split(original_ids,test_size=.2,random_state=42)
train_ids,val_ids=train_test_split(cnn_train_ids,test_size=.2,random_state=123,stratify=labels.loc[np.asarray(cnn_train_ids,dtype=int)])
train_ids=set(map(int,train_ids)); val_ids=set(map(int,val_ids)); test_ids=set(map(int,cnn_test_ids))
assert not (train_ids&val_ids or train_ids&test_ids or val_ids&test_ids)
print('Original CNN split: train',len(cnn_train_ids),'test',len(cnn_test_ids))
print('Fusion train/val/test:',len(train_ids),len(val_ids),len(test_ids))
print('Patient ID overlaps:',len(train_ids&test_ids),len(val_ids&test_ids))
# Check whether saved image features were derived from this checkpoint by recomputing them.
ckpt=OUT/'cnn_roi_model.pth'
assert ckpt.exists(),f'Missing checkpoint: {ckpt}'
net=models.resnet18(weights=None); net.fc=nn.Linear(net.fc.in_features,2)
state=torch.load(ckpt,map_location='cpu',weights_only=True)
net.load_state_dict(state,strict=True)
net.eval()
feature_net=nn.Sequential(*list(net.children())[:-1]).eval()
tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
img_dir=Path('data/processed/images_roi')
features={}; probs={}
print('Extracting ROI CNN embeddings from saved checkpoint (CPU may take time)...',flush=True)
with torch.inference_mode():
    for n,(pid,group) in enumerate(manifest.groupby('pid',sort=False),1):
        batch=[]
        for name in group['image_file']:
            path=img_dir/str(name)
            if not path.exists(): raise FileNotFoundError(path)
            with Image.open(path) as im: batch.append(tfm(im.convert('RGB')))
        vectors=[]; p=[]
        for j in range(0,len(batch),16):
            x=torch.stack(batch[j:j+16]); vectors.append(feature_net(x).flatten(1).numpy()); p.append(torch.softmax(net(x),dim=1)[:,1].numpy())
        features[pid]=np.concatenate(vectors).mean(axis=0)
        probs[pid]=float(np.concatenate(p).mean())
        if n%100==0: print(f'Extracted {n}/{manifest.pid.nunique()} patients',flush=True)

# Strictly exclude diagnosis-derived, post-outcome and identifier columns.
exclude={'Diagnosis','Diagnosis_Presumptive','Management','Severity','Peritonitis','Length_of_Stay','Perforation','Appendicular_Abscess','Abscess_Location','US_Number','US_Number_int','has_images','pid'}
exclude.update(c for c in df if c.startswith('img_feat_'))
clinical_cols=[c for c in df if c not in exclude]
clinical=df.set_index('pid')
# Guard against unexpected target leakage columns.
assert not set(clinical_cols)&exclude
num=[c for c in clinical_cols if pd.api.types.is_numeric_dtype(clinical[c])]
cat=[c for c in clinical_cols if c not in num]
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],sparse_threshold=0)
# Preserve row order explicitly.
tr=sorted(train_ids); va=sorted(val_ids); te=sorted(test_ids)
Xt=pre.fit_transform(clinical.loc[tr,clinical_cols]); Xv=pre.transform(clinical.loc[va,clinical_cols]); Xe=pre.transform(clinical.loc[te,clinical_cols]);
sc=StandardScaler(); It=sc.fit_transform(np.stack([features[i] for i in tr])); Iv=sc.transform(np.stack([features[i] for i in va])); Ie=sc.transform(np.stack([features[i] for i in te]));
yt=labels.loc[tr].to_numpy(dtype=np.int64); yv=labels.loc[va].to_numpy(dtype=np.int64); ye=labels.loc[te].to_numpy(dtype=np.int64)
# Small gated residual fusion: starts near clinical-only; gates image contribution.
class GatedFusion(nn.Module):
    def __init__(self,n_tab,n_img):
        super().__init__()
        self.clin=nn.Sequential(nn.Linear(n_tab,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
        self.image=nn.Sequential(nn.Linear(n_img,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
        self.gate=nn.Sequential(nn.Linear(n_tab+n_img,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.constant_(self.gate[-1].bias,-3.0)
    def forward(self,t,i,mode):
        c=self.clin(t).squeeze(-1)
        if mode=='clinical': return c
        im=self.image(i).squeeze(-1)
        if mode=='image': return im
        g=torch.sigmoid(self.gate(torch.cat([t,i],dim=1))).squeeze(-1)
        return c+g*im

def fit(mode):
    torch.manual_seed(SEED)
    m=GatedFusion(Xt.shape[1],It.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    t=torch.tensor(Xt,dtype=torch.float32); i=torch.tensor(It,dtype=torch.float32); y=torch.tensor(yt,dtype=torch.float32)
    vt=torch.tensor(Xv,dtype=torch.float32); vi=torch.tensor(Iv,dtype=torch.float32)
    best=None; bestloss=float('inf'); patience=0; bestepoch=0
    for epoch in range(150):
        m.train()
        order=torch.randperm(len(y))
        for ix in order.split(64):
            opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(m(t[ix],i[ix],mode),y[ix]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.functional.binary_cross_entropy_with_logits(m(vt,vi,mode),torch.tensor(yv,dtype=torch.float32)).item()
        if vl<bestloss-1e-4: bestloss=vl; best=copy.deepcopy(m.state_dict()); bestepoch=epoch+1; patience=0
        else: patience+=1
        if patience>=15: break
    m.load_state_dict(best); m.eval()
    with torch.no_grad(): p=torch.sigmoid(m(torch.tensor(Xe,dtype=torch.float32),torch.tensor(Ie,dtype=torch.float32),mode)).numpy()
    return m,p,bestepoch

results=[]; predictions={}
for mode in ('clinical','image','fusion'):
    model,p,epochs=fit(mode); predictions[mode]=p
    pred=(p>=.5).astype(int)
    record={'model':mode,'accuracy':accuracy_score(ye,pred),'macro_f1':f1_score(ye,pred,average='macro'),'auc':roc_auc_score(ye,p),'test_n':len(te),'epochs':epochs}
    results.append(record); print(record,flush=True)
    torch.save(model.state_dict(),OUT/f'roi_checkpoint_{mode}_fusion_exploratory.pth')
# Original ROI CNN direct patient prediction, from exact saved checkpoint.
roi_p=np.array([probs[i] for i in te]); roi_pred=(roi_p>=.5).astype(int)
record={'model':'original_roi_cnn','accuracy':accuracy_score(ye,roi_pred),'macro_f1':f1_score(ye,roi_pred,average='macro'),'auc':roc_auc_score(ye,roi_p),'test_n':len(te),'epochs':0}
results.append(record); predictions['original_roi_cnn']=roi_p; print(record)
pd.DataFrame(results).to_csv(OUT/'roi_checkpoint_fusion_exploratory_results.csv',index=False)
pd.DataFrame({'patient_id':te,'true_label':ye,**{k+'_prob':v for k,v in predictions.items()}}).to_csv(OUT/'roi_checkpoint_fusion_test_predictions.csv',index=False)
plt.figure(figsize=(7,6))
for name,p in predictions.items():
    fpr,tpr,_=roc_curve(ye,p); plt.plot(fpr,tpr,label=f'{name} (AUC {roc_auc_score(ye,p):.3f})')
plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('Exploratory patient-holdout ROC (appendicitis positive)'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'figures'/'roc_roi_checkpoint_fusion_exploratory.png',dpi=180)
print('\nSaved outputs/roi_checkpoint_fusion_exploratory_results.csv')
print('Saved outputs/figures/roc_roi_checkpoint_fusion_exploratory.png')
print('CAUTION: CNN checkpoint provenance and image preprocessing timing not independently verified. This is an exploratory holdout, not external validation.')
