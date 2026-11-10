import re
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

IMG_DIR = Path("data/raw/images/US_Pictures")
OUT_DIR = Path("data/processed/images_resized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

IMG_SIZE = (224, 224)  # standard CNN input size
resized_count, error_count = 0, 0
manifest = []

for _, row in df.iterrows():
    us_num = row.get("US_Number_int")
    if pd.isna(us_num) or us_num not in patient_imgs:
        continue
    pid = int(us_num)
    diagnosis = row["Diagnosis"]

    for img_path in patient_imgs[pid]:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            out_name = f"{pid}_{img_path.stem}.png"
            out_path = OUT_DIR / out_name
            img.save(out_path)
            manifest.append({
                "patient_id": pid,
                "image_file": out_name,
                "diagnosis": diagnosis,
                "severity": row.get("Severity"),
            })
            resized_count += 1
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            error_count += 1

print(f"\nTotal images resized: {resized_count}")
print(f"Errors: {error_count}")

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv("data/processed/image_manifest.csv", index=False)
print(f"\nSaved manifest: data/processed/image_manifest.csv")
print(f"Manifest shape: {manifest_df.shape}")
print(f"\nUnique patients in manifest: {manifest_df['patient_id'].nunique()}")
print(f"\nDiagnosis distribution in manifest:\n{manifest_df['diagnosis'].value_counts()}")