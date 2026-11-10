# FYP Results: Base Paper vs Our Model
## Multimodal, Feature-Optimized, Uncertainty-Aware Deep Learning Framework
### for Pediatric Appendicitis & Peritonitis Diagnosis

---

## Target 1: Diagnosis (Appendicitis vs No Appendicitis)

| Method | Accuracy | F1-Score | AUC | Features Used |
|---|---|---|---|---|
| **Base Paper** (Dyuthishree et al., 2026) | 93.24% | — | — | 58 |
| Our Model — Baseline (XGBoost, all features) | 96.79% | 0.9606 | 0.9879 | 52 |
| Our Model — Boruta-selected features | 95.51% | 0.9449 | 0.9812 | 15 |
| Our Model — Baseline + SMOTE | 96.79% | 0.9606 | 0.9865 | 52 |
| **Our Model — Duck Swarm Algorithm (DSA)-selected** | **96.79%** | **0.9612** | **0.9886** | 30 |

**Improvement over Base Paper: +3.55 percentage points**

---

## Target 2: Severity (Complicated vs Uncomplicated — Peritonitis-type)

| Method | Accuracy | F1-Score | AUC |
|---|---|---|---|
| **Base Paper** (Dyuthishree et al., 2026) | 88.17% | — | — |
| Our Model — Baseline (XGBoost, all features) | 92.99% | 0.9585 | 0.9762 |
| Our Model — Baseline + SMOTE | 91.72% | 0.9498 | 0.9762 |
| **Our Model — 5-Fold Cross-Validation Mean** | **93.47% (± 1.87%)** | — | — |

**Improvement over Base Paper: +5.30 percentage points**

---

## Key Observations

1. Both diagnostic targets (Diagnosis and Severity) outperform the base paper using a simple XGBoost baseline, before any deep learning fusion or uncertainty modeling was applied.
2. The Duck Swarm Algorithm (DSA) achieved the best overall trade-off: same accuracy as the full-feature baseline, slightly higher AUC, and 42% fewer features (30 vs 52).
3. Boruta produced the most aggressive feature reduction (15 features), with a modest drop in accuracy — useful for a lightweight model variant.
4. SMOTE did not provide a measurable improvement on this dataset, suggesting XGBoost's native handling of class imbalance was sufficient.
5. 5-fold cross-validation on the Severity target confirmed result stability (93.47% ± 1.87%), ruling out a lucky train/test split.

*Note: Results above use tabular clinical data only. Multimodal fusion with ultrasound image features (CNN) is expected to further improve performance and is reported separately once completed.*