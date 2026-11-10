import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

IMG_DIR = Path("data/processed/images_resized")
manifest = pd.read_csv("data/processed/image_manifest.csv")
manifest["label"] = (manifest["diagnosis"] == "appendicitis").astype(int)

patients = manifest["patient_id"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
print(f"Test images: {len(test_df)} ({test_df['patient_id'].nunique()} patients)")

class USDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image_file"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.long)
        return img, label

imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
vit_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ============================================================
# LOAD THE 3 TRAINED MODELS
# ============================================================
print("Loading trained models...")

cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(torch.load("outputs/cnn_model.pth", map_location=device))
cnn_model.eval()

vit_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
vit_model.load_state_dict(torch.load("outputs/vit_model.pth", map_location=device))
vit_model.eval()

eff_model = models.efficientnet_b0(weights=None)
eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 2)
eff_model.load_state_dict(torch.load("outputs/efficientnet_model.pth", map_location=device))
eff_model.eval()

# ============================================================
# GET PREDICTIONS FROM EACH MODEL ON TEST SET
# ============================================================
cnn_ds = USDataset(test_df, IMG_DIR, imagenet_transform)
vit_ds = USDataset(test_df, IMG_DIR, vit_transform)

cnn_loader = DataLoader(cnn_ds, batch_size=16, shuffle=False)
vit_loader = DataLoader(vit_ds, batch_size=16, shuffle=False)
eff_loader = DataLoader(cnn_ds, batch_size=16, shuffle=False)  # EfficientNet uses same normalization as CNN

def get_probs(model, loader):
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs)
            p = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.numpy())
    return np.array(probs)

print("Getting CNN predictions...")
cnn_probs = get_probs(cnn_model, cnn_loader)
print("Getting ViT predictions...")
vit_probs = get_probs(vit_model, vit_loader)
print("Getting EfficientNet predictions...")
eff_probs = get_probs(eff_model, eff_loader)

labels = test_df["label"].values

# ============================================================
# ENSEMBLE: AVERAGE THE 3 MODELS' PROBABILITIES
# ============================================================
ensemble_probs = (cnn_probs + vit_probs + eff_probs) / 3
ensemble_preds = (ensemble_probs > 0.5).astype(int)

acc = accuracy_score(labels, ensemble_preds)
f1 = f1_score(labels, ensemble_preds)
auc = roc_auc_score(labels, ensemble_probs)
print(f"\n[ENSEMBLE - Image level] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Patient-level
test_df = test_df.copy()
test_df["ensemble_prob"] = ensemble_probs
patient_results = test_df.groupby("patient_id").agg(
    label=("label", "first"), avg_prob=("ensemble_prob", "mean")
).reset_index()
patient_results["pred"] = (patient_results["avg_prob"] > 0.5).astype(int)

p_acc = accuracy_score(patient_results["label"], patient_results["pred"])
p_f1 = f1_score(patient_results["label"], patient_results["pred"])
p_auc = roc_auc_score(patient_results["label"], patient_results["avg_prob"])
print(f"[ENSEMBLE - Patient level, averaged] Accuracy: {p_acc:.4f} | F1: {p_f1:.4f} | AUC: {p_auc:.4f}")

results = pd.DataFrame([
    {"name": "Ensemble (CNN+ViT+EfficientNet, image-level)", "accuracy": acc, "f1": f1, "auc": auc},
    {"name": "Ensemble (patient-level, avg)", "accuracy": p_acc, "f1": p_f1, "auc": p_auc},
])
results.to_csv("outputs/phase3_ensemble_results.csv", index=False)
print("\nSaved: outputs/phase3_ensemble_results.csv")