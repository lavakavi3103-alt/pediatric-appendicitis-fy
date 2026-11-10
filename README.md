# Multimodal, Feature-Optimized and Uncertainty-Aware Deep Learning Framework for Secure Pediatric Appendicitis & Peritonitis Diagnosis

## Overview

This project proposes a multimodal diagnostic framework for pediatric appendicitis detection and severity (peritonitis) classification, combining structured clinical data with ultrasound imaging. It builds on and extends prior work (Dyuthishree et al.) by introducing comparative feature selection (Boruta vs. a custom Duck Swarm Algorithm), multimodal fusion, conformal prediction for calibrated uncertainty, and a secure, deployable architecture.

## Dataset

**Regensburg Pediatric Appendicitis Dataset**
- 782 patients, 58 clinical attributes
- 2,097 ultrasound images (1–19 images per patient, 578 patients with matched images)
- Targets: Diagnosis (appendicitis / no appendicitis), Severity (complicated / uncomplicated)
- Source: UCI Machine Learning Repository

## Project Pipeline

| Phase | Description | Status |
|---|---|---|
| 1 | Data exploration, patient–image linking | Complete |
| 2 | Tabular baseline (XGBoost), Boruta, Duck Swarm Algorithm (DSA), SMOTE | Complete (Diagnosis); partial (Severity) |
| 3 | Image models: ResNet-18 CNN, ROI cropping, ViT, EfficientNet, ensembling | Complete |
| 4 | Multimodal fusion (tabular + image features) | Complete (Diagnosis); pending (Severity) |
| 5 | Uncertainty quantification (conformal prediction), explainability (SHAP, Grad-CAM) | In progress |
| 6 | Security (ChaCha20-Poly1305 encryption), deployment (FastAPI + web frontend) | Planned |

## Key Results

### Diagnosis (Appendicitis vs. No Appendicitis)

| Method | Accuracy | F1 | AUC |
|---|---|---|---|
| Base paper (Dyuthishree et al.) | 93.24% | — | — |
| Baseline (XGBoost, all features) | 96.79% | 0.961 | 0.988 |
| Boruta (15 features) | 95.51% | 0.945 | 0.981 |
| **Duck Swarm Algorithm (30 features)** | **96.79%** | **0.961** | **0.989** |

### Severity (Complicated / Peritonitis)

| Method | Accuracy | AUC |
|---|---|---|
| Base paper (Dyuthishree et al.) | 88.17% | — |
| Baseline (XGBoost) | 92.99% | 0.976 |
| 5-fold cross-validation (mean) | 93.47% (± 1.87%) | — |

### Image Models (Diagnosis, patient-level)

| Model | Accuracy | AUC |
|---|---|---|
| ResNet-18 (plain resize) | 68.10% | 0.669 |
| Vision Transformer (ViT-Tiny) | 63.79% | 0.627 |
| EfficientNet-B0 | 64.66% | 0.692 |
| Ensemble (3 models) | 68.97% | 0.723 |
| **ResNet-18 + ROI cropping** | **76.72%** | **0.805** |

### Multimodal Fusion (578-patient subset)

| Configuration | Accuracy | F1 | AUC |
|---|---|---|---|
| Tabular-only (DSA features) | 94.83% | 0.923 | 0.993 |
| Image-only (CNN features) | 69.83% | 0.462 | 0.768 |
| Fusion (concatenated) | 93.97% | 0.909 | 0.992 |

### Uncertainty Quantification (Split Conformal Prediction)

| Metric | Value |
|---|---|
| Target coverage | 90% |
| Empirical coverage | 92.24% |
| Singleton (confident) predictions | 99.14% |
| Accuracy on flagged (low-confidence) cases | 100% |

## Project Structure