import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.manual_seed(42)
device = torch.device("cpu")

IMG_DIR = Path("data/processed/images_resized")
manifest = pd.read_csv("data/processed/image_manifest.csv")
manifest["label"] = (manifest["diagnosis"] == "appendicitis").astype(int)

patients = manifest["patient_id"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
print(f"Test images: {len(test_df)} ({test_df['patient_id'].nunique()} patients)")

# Load best model (CNN - ResNet18)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("outputs/cnn_model.pth", map_location=device))
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 4 versions: original, horizontal flip, 10 deg rotation, slight zoom (center crop)
def get_versions(img):
    versions = []
    versions.append(transforms.ToTensor()(img))
    versions.append(transforms.ToTensor()(transforms.functional.hflip(img)))
    versions.append(transforms.ToTensor()(transforms.functional.rotate(img, 10)))
    cropped = transforms.functional.center_crop(img, (190, 190))
    cropped = transforms.functional.resize(cropped, (224, 224))
    versions.append(transforms.ToTensor()(cropped))
    return [normalize(v) for v in versions]

print("Running Test-Time Augmentation (4 views per image)...")
all_probs, all_labels = [], []
with torch.no_grad():
    for i, row in test_df.iterrows():
        img = Image.open(IMG_DIR / row["image_file"]).convert("RGB")
        versions = get_versions(img)
        batch = torch.stack(versions)
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        avg_prob = probs.mean().item()
        all_probs.append(avg_prob)
        all_labels.append(row["label"])
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_df)}")

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = (all_probs > 0.5).astype(int)

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
print(f"\n[CNN + TTA - Image level] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

test_df = test_df.copy()
test_df["tta_prob"] = all_probs
patient_results = test_df.groupby("patient_id").agg(
    label=("label", "first"), avg_prob=("tta_prob", "mean")
).reset_index()
patient_results["pred"] = (patient_results["avg_prob"] > 0.5).astype(int)

p_acc = accuracy_score(patient_results["label"], patient_results["pred"])
p_f1 = f1_score(patient_results["label"], patient_results["pred"])
p_auc = roc_auc_score(patient_results["label"], patient_results["avg_prob"])
print(f"[CNN + TTA - Patient level, averaged] Accuracy: {p_acc:.4f} | F1: {p_f1:.4f} | AUC: {p_auc:.4f}")

results = pd.DataFrame([
    {"name": "CNN + TTA (image-level)", "accuracy": acc, "f1": f1, "auc": auc},
    {"name": "CNN + TTA (patient-level)", "accuracy": p_acc, "f1": p_f1, "auc": p_auc},
])
results.to_csv("outputs/phase3_cnn_tta_results.csv", index=False)
print("\nSaved: outputs/phase3_cnn_tta_results.csv")