"""Exploratory conformal sets for 11e DSA ROI fusion; validation reused for early stopping.
Place alongside 11d_roi_cnn_safe_fusion.py. Does not overwrite 11e outputs.
"""
from pathlib import Path
base = Path(__file__).with_name('11d_roi_cnn_safe_fusion.py')
if not base.exists():
    raise FileNotFoundError(f'Missing {base}')
src = base.read_text(encoding='utf-8')
needle = '# Small gated residual fusion: starts near clinical-only; gates image contribution.'
if src.count(needle) != 1:
    raise RuntimeError('Expected insertion point missing in 11d')
# Reuse exactly the feature selection logic of 11e.
dsa = Path(__file__).with_name('11e_dsa_roi_fusion.py')
if not dsa.exists():
    raise FileNotFoundError(f'Missing {dsa}')
import ast
code = dsa.read_text(encoding='utf-8')
module = ast.parse(code)
insert = None
for node in ast.walk(module):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'insert' for t in node.targets):
        insert = ast.literal_eval(node.value)
        break
if insert is None:
    raise RuntimeError('Cannot find DSA insertion in 11e')
src = src.replace(needle, insert + '\n' + needle)
# Add calibration after models are trained; use the fusion model's validation predictions.
marker = "pd.DataFrame(results).to_csv(OUT/'roi_checkpoint_fusion_exploratory_results.csv',index=False)"
if src.count(marker) != 1:
    raise RuntimeError('Cannot find output insertion point')
extra = '''# EXPLORATORY ONLY: validation data were used for early stopping, so calibration is not independent.
from math import ceil
fusion_net=GatedFusion(Xt.shape[1], It.shape[1])
fusion_net.load_state_dict(torch.load(OUT/'dsa_roi_fusion_fusion_exploratory.pth',map_location='cpu',weights_only=True))
fusion_net.eval()
with torch.inference_mode():
    val_probs=torch.sigmoid(fusion_net(torch.tensor(Xv,dtype=torch.float32),torch.tensor(Iv,dtype=torch.float32),'fusion')).numpy()
test_probs=predictions['fusion']
alpha=0.10
cal_scores=np.where(yv==1,1-val_probs,val_probs)
q_index=min(len(cal_scores),ceil((len(cal_scores)+1)*(1-alpha)))
qhat=float(np.sort(cal_scores)[q_index-1])
include_no=(test_probs<=qhat)
include_yes=((1-test_probs)<=qhat)
set_size=include_no.astype(int)+include_yes.astype(int)
covered=np.where(ye==1,include_yes,include_no)
summary={'method':'exploratory_split_conformal_validation_reused_for_early_stopping',
         'alpha':alpha,'nominal_coverage':1-alpha,'calibration_n':len(yv),'test_n':len(ye),
         'qhat':qhat,'empirical_coverage':float(covered.mean()),
         'average_set_size':float(set_size.mean()),'singleton_n':int((set_size==1).sum()),
         'ambiguous_n':int((set_size==2).sum()),'empty_n':int((set_size==0).sum())}
print('EXPLORATORY UNCERTAINTY:',summary,flush=True)
pd.DataFrame([summary]).to_csv(OUT/'dsa_roi_fusion_uncertainty_exploratory.csv',index=False)
pd.DataFrame({'patient_id':te,'true_label':ye,'fusion_probability_appendicitis':test_probs,
              'include_no_appendicitis':include_no,'include_appendicitis':include_yes,
              'prediction_set_size':set_size,'covered':covered}).to_csv(
              OUT/'dsa_roi_fusion_uncertainty_sets_exploratory.csv',index=False)
print('WARNING: Validation set was reused for early stopping and conformal calibration. Coverage is NOT formally guaranteed.')
'''
src=src.replace(marker,extra+'\n'+marker)
src=src.replace('roi_checkpoint_fusion_exploratory_results.csv','dsa_roi_fusion_exploratory_results.csv')
src=src.replace('roi_checkpoint_fusion_test_predictions.csv','dsa_roi_fusion_test_predictions.csv')
src=src.replace('roc_roi_checkpoint_fusion_exploratory.png','roc_dsa_roi_fusion_exploratory.png')
src=src.replace("OUT/f'roi_checkpoint_{mode}_fusion_exploratory.pth'", "OUT/f'dsa_roi_{mode}_fusion_exploratory.pth'")
print('Running DSA fusion + exploratory conformal uncertainty...',flush=True)
exec(compile(src,str(base),'exec'),{'__name__':'__main__','__file__':str(base)})


