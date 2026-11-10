import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
IMG_DIR = Path("data/processed/images_resized")

manifest = pd.read_csv("data/processed/image_manifest.csv")
manifest["label"] = (manifest["diagnosis"] == "appendicitis").astype(int)

patients = manifest["patient_id"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
train_df = manifest[manifest["patient_id"].isin(train_patients)].reset_index(drop=True)
test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
print(f"Train images: {len(train_df)} ({train_df['patient_id'].nunique()} patients)")
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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = USDataset(train_df, IMG_DIR, train_transform)
test_ds = USDataset(test_df, IMG_DIR, test_transform)

class_counts = train_df["label"].value_counts().sort_index()
sample_weights = train_df["label"].map(lambda l: 1.0 / class_counts[l]).values
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

device = torch.device("cpu")
print(f"Using device: {device}")

print("\nLoading pretrained EfficientNet-B0...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if "features.8" in name or "features.7" in name or "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)

EPOCHS = 12
print(f"\nTraining EfficientNet-B0 ({EPOCHS} epochs, CPU)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
print(f"\n[EfficientNet-B0 - Image level] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

test_df = test_df.copy()
test_df["pred_prob"] = all_probs
patient_results = test_df.groupby("patient_id").agg(
    label=("label", "first"), avg_prob=("pred_prob", "mean")
).reset_index()
patient_results["pred"] = (patient_results["avg_prob"] > 0.5).astype(int)

p_acc = accuracy_score(patient_results["label"], patient_results["pred"])
p_f1 = f1_score(patient_results["label"], patient_results["pred"])
p_auc = roc_auc_score(patient_results["label"], patient_results["avg_prob"])
print(f"[EfficientNet-B0 - Patient level, averaged] Accuracy: {p_acc:.4f} | F1: {p_f1:.4f} | AUC: {p_auc:.4f}")

torch.save(model.state_dict(), OUT / "efficientnet_model.pth")
print("\nSaved model: outputs/efficientnet_model.pth")

results = pd.DataFrame([
    {"name": "EfficientNet-B0 (image-level)", "accuracy": acc, "f1": f1, "auc": auc},
    {"name": "EfficientNet-B0 (patient-level, avg)", "accuracy": p_acc, "f1": p_f1, "auc": p_auc},
])
results.to_csv(OUT / "phase3_efficientnet_results.csv", index=False)
print("Saved: outputs/phase3_efficientnet_results.csv")