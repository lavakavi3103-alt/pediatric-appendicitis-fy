import warnings
warnings.filterwarnings("ignore")

import re
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

# ============================================================
# Re-crop with a tighter margin (10% instead of 5%)
# ============================================================
RAW_IMG_DIR = Path("data/raw/images/US_Pictures")
ROI_DIR = Path("data/processed/images_roi_v2")
ROI_DIR.mkdir(parents=True, exist_ok=True)

df_tab = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
df_tab = df_tab[df_tab["Diagnosis"].notna()].copy()
df_tab["US_Number_int"] = df_tab["US_Number"].dropna().astype(int)

pattern = re.compile(r"^(\d+)\.(\d+)")
patient_imgs = {}
for p in RAW_IMG_DIR.iterdir():
    if p.is_file():
        m = pattern.match(p.name)
        if m:
            pid = int(m.group(1))
            patient_imgs.setdefault(pid, []).append(p)

def auto_crop_roi_v2(img):
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    threshold = 15
    rows_with_content = np.where(gray.max(axis=1) > threshold)[0]
    cols_with_content = np.where(gray.max(axis=0) > threshold)[0]
    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return img
    top, bottom = rows_with_content[0], rows_with_content[-1]
    left, right = cols_with_content[0], cols_with_content[-1]
    # Tighter margin (10%) - scan markers innum konjam adhigam remove pannuvom
    margin_h = int((bottom - top) * 0.10)
    margin_w = int((right - left) * 0.10)
    top = max(0, top + margin_h)
    bottom = min(h, bottom - margin_h)
    left = max(0, left + margin_w)
    right = min(w, right - margin_w)
    if bottom <= top or right <= left:
        return img
    return img.crop((left, top, right, bottom))

print("Re-cropping images with tighter ROI margin...")
manifest = []
for _, row in df_tab.iterrows():
    us_num = row.get("US_Number_int")
    if pd.isna(us_num) or us_num not in patient_imgs:
        continue
    pid = int(us_num)
    for img_path in patient_imgs[pid]:
        try:
            img = Image.open(img_path).convert("RGB")
            roi_img = auto_crop_roi_v2(img)
            roi_img = roi_img.resize((224, 224))
            out_name = f"{pid}_{img_path.stem}.png"
            roi_img.save(ROI_DIR / out_name)
            manifest.append({
                "patient_id": pid, "image_file": out_name,
                "diagnosis": row["Diagnosis"], "severity": row.get("Severity"),
            })
        except Exception as e:
            print(f"Error: {img_path.name}: {e}")

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv("data/processed/image_manifest_roi_v2.csv", index=False)
print(f"Re-cropped {len(manifest_df)} images.")

# ============================================================
# TRAINING
# ============================================================
manifest = manifest_df
manifest["label"] = (manifest["diagnosis"].str.lower() == "appendicitis").astype(int)
print(f"\nTotal images: {len(manifest)} | Patients: {manifest['patient_id'].nunique()}")

patients = manifest["patient_id"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
train_df = manifest[manifest["patient_id"].isin(train_patients)].reset_index(drop=True)
test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
print(f"Train images: {len(train_df)} ({train_df['patient_id'].nunique()} patients)")
print(f"Test images: {len(test_df)} ({test_df['patient_id'].nunique()} patients)")

class USDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image_file"]).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(row["label"], dtype=torch.long)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = USDataset(train_df, ROI_DIR, train_transform)
test_ds = USDataset(test_df, ROI_DIR, test_transform)

class_counts = train_df["label"].value_counts().sort_index()
sample_weights = train_df["label"].map(lambda l: 1.0 / class_counts[l]).values
sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

device = torch.device("cpu")
print(f"\nUsing device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = True   # full fine-tune
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

EPOCHS = 35
print("\n" + "=" * 60)
print(f"Training ROI CNN v2 ({EPOCHS} epochs)")
print("=" * 60)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

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
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)
print(f"\n[ROI CNN v2 - Image level] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

test_results = test_df.copy()
test_results["pred_prob"] = all_probs
patient_results = test_results.groupby("patient_id").agg(
    label=("label", "first"), avg_prob=("pred_prob", "mean")
).reset_index()
patient_results["pred"] = (patient_results["avg_prob"] > 0.5).astype(int)

p_acc = accuracy_score(patient_results["label"], patient_results["pred"])
p_f1 = f1_score(patient_results["label"], patient_results["pred"], zero_division=0)
p_auc = roc_auc_score(patient_results["label"], patient_results["avg_prob"])
print(f"[ROI CNN v2 - Patient level] Accuracy: {p_acc:.4f} | F1: {p_f1:.4f} | AUC: {p_auc:.4f}")

torch.save(model.state_dict(), OUT / "cnn_roi_v2_model.pth")
print(f"\nSaved model: outputs/cnn_roi_v2_model.pth")

results = pd.DataFrame([
    {"name": "ROI CNN v2 (image-level)", "accuracy": acc, "f1": f1, "auc": auc},
    {"name": "ROI CNN v2 (patient-level)", "accuracy": p_acc, "f1": p_f1, "auc": p_auc},
])
results.to_csv(OUT / "phase3_roi_cnn_v2_results.csv", index=False)
print("Saved: outputs/phase3_roi_cnn_v2_results.csv")