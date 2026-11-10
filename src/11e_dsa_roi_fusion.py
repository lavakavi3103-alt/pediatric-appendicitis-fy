"""Run train-only DSA-inspired feature selection with the existing 11d ROI CNN fusion.

Place this file alongside 11d_roi_cnn_safe_fusion.py in src/.
No held-out test labels are accessed during feature selection.
"""
from pathlib import Path
import runpy

base = Path(__file__).with_name('11d_roi_cnn_safe_fusion.py')
if not base.exists():
    raise FileNotFoundError(f'Place 11d_roi_cnn_safe_fusion.py beside this file: {base}')
src = base.read_text(encoding='utf-8')
needle = "# Small gated residual fusion: starts near clinical-only; gates image contribution."
if src.count(needle) != 1:
    raise RuntimeError('11d script has changed; DSA insertion point not found uniquely')
insert = '''# DSA-inspired binary swarm search, strictly on an INNER split of training patients.
# No outer validation/test rows used in feature selection.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split as _split

print('DSA-inspired train-only clinical feature selection...', flush=True)
_rng = np.random.default_rng(SEED)
_inner_tr, _inner_va = _split(np.arange(len(yt)), test_size=0.25,
                              random_state=SEED, stratify=yt)
_nfeat = Xt.shape[1]
_n_ducks, _n_steps = 10, 12
_ducks = _rng.random((_n_ducks, _nfeat)) < 0.55
_best_mask = None
_best_score = float('inf')
_cache = {}

def _fitness(mask):
    if not mask.any():
        return 1e6
    key = np.packbits(mask).tobytes()
    if key in _cache:
        return _cache[key]
    clf = LogisticRegression(max_iter=350, solver='liblinear', random_state=SEED)
    clf.fit(Xt[_inner_tr][:, mask], yt[_inner_tr])
    p = clf.predict_proba(Xt[_inner_va][:, mask])[:, 1]
    score = log_loss(yt[_inner_va], p, labels=[0, 1]) + 0.035 * mask.mean()
    _cache[key] = float(score)
    return float(score)

for _step in range(_n_steps):
    for _duck in _ducks:
        _score = _fitness(_duck)
        if _score < _best_score:
            _best_score, _best_mask = _score, _duck.copy()
    # Attraction toward best duck + stochastic local exploration.
    # This is a DSA-inspired binary swarm heuristic, not a canonical DSA reproduction.
    _new = []
    for _duck in _ducks:
        _candidate = _duck.copy()
        _follow = _rng.random(_nfeat) < 0.18
        _candidate[_follow] = _best_mask[_follow]
        _mutate = _rng.random(_nfeat) < (0.10 * (1 - _step / _n_steps) + 0.015)
        _candidate[_mutate] = ~_candidate[_mutate]
        if not _candidate.any():
            _candidate[_rng.integers(_nfeat)] = True
        _new.append(_candidate)
    _ducks = np.array(_new, dtype=bool)

_selected = np.flatnonzero(_best_mask)
print(f'DSA-inspired selected {_selected.size}/{_nfeat} encoded clinical features; inner score={_best_score:.4f}', flush=True)
pd.DataFrame({'encoded_feature_index': _selected}).to_csv(
    OUT/'dsa_roi_selected_encoded_features.csv', index=False)
Xt = Xt[:, _selected].copy()
Xv = Xv[:, _selected].copy()
Xe = Xe[:, _selected].copy()
'''
src = src.replace(needle, insert + '\n' + needle)
# Distinct outputs: preserve prior 11d run.
src = src.replace('roi_checkpoint_fusion_exploratory_results.csv', 'dsa_roi_fusion_exploratory_results.csv')
src = src.replace('roi_checkpoint_fusion_test_predictions.csv', 'dsa_roi_fusion_test_predictions.csv')
src = src.replace('roc_roi_checkpoint_fusion_exploratory.png', 'roc_dsa_roi_fusion_exploratory.png')
src = src.replace("OUT/f'roi_checkpoint_{mode}_fusion_exploratory.pth'", "OUT/f'dsa_roi_{mode}_fusion_exploratory.pth'")
src = src.replace("'Exploratory patient-holdout ROC (appendicitis positive)'", "'DSA-inspired ROI fusion: exploratory ROC (appendicitis positive)'")
print('Running DSA-inspired ROI CNN gated fusion. Existing checkpoint will not be retrained.', flush=True)
exec(compile(src, str(base), 'exec'), {'__name__': '__main__', '__file__': str(base)})
