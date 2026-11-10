import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = Path("data/raw/images/US_Pictures")

df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
df = df[df["Diagnosis"].notna()].copy()
df["US_Number_int"] = df["US_Number"].dropna().astype(int)

pattern = re.compile(r"^(\d+)\.(\d+)")
patient_imgs = {}
for p in IMG_DIR.iterdir():
    if p.is_file():
        m = pattern.match(p.name)
        if m:
            pid = int(m.group(1))
            patient_imgs.setdefault(pid, []).append(p)

df["has_images"] = df["US_Number_int"].apply(
    lambda x: (x in patient_imgs) if pd.notna(x) else False
)
multimodal_df = df[df["has_images"]].reset_index(drop=True)
print("Multimodal patients:", len(multimodal_df))

# ============================================================
# Same ROI cropping logic as 09g_roi_preprocess.py
# ============================================================
def auto_crop_roi(img):
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    threshold = 15
    rows_with_content = np.where(gray.max(axis=1) > threshold)[0]
    cols_with_content = np.where(gray.max(axis=0) > threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return img

    top, bottom = rows_with_content[0], rows_with_content[-1]
    left, right = cols_with_content[0], cols_with_content[-1]

    margin_h = int((bottom - top) * 0.05)
    margin_w = int((right - left) * 0.05)
    top = max(0, top + margin_h)
    bottom = min(h, bottom - margin_h)
    left = max(0, left + margin_w)
    right = min(w, right - margin_w)

    if bottom <= top or right <= left:
        return img

    return img.crop((left, top, right, bottom))

# ============================================================
# Load ROI-trained CNN, remove final layer -> feature extractor
# ============================================================
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("outputs/cnn_roi_model.pth", map_location=device))
model.fc = nn.Identity()
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\nExtracting ROI-CNN features for each patient's images...")
image_features = {}
with torch.no_grad():
    for i, pid in enumerate(multimodal_df["US_Number_int"]):
        imgs = patient_imgs[pid]
        feats = []
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            img = auto_crop_roi(img)
            img = img.resize((224, 224))
            img_tensor = transform(img).unsqueeze(0).to(device)
            feat = model(img_tensor).squeeze().numpy()
            feats.append(feat)
        image_features[pid] = np.mean(feats, axis=0)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(multimodal_df)} patients")

print("Done extracting ROI image features.")

img_feat_cols = [f"img_feat_{i}" for i in range(512)]
img_feat_df = pd.DataFrame(
    [image_features[pid] for pid in multimodal_df["US_Number_int"]],
    columns=img_feat_cols
)

fusion_df = pd.concat([multimodal_df.reset_index(drop=True), img_feat_df], axis=1)
fusion_df.to_csv(OUT / "fusion_dataset_roi.csv", index=False)
print(f"\nSaved: data/processed/fusion_dataset_roi.csv")
print(f"Fusion dataset shape: {fusion_df.shape}")