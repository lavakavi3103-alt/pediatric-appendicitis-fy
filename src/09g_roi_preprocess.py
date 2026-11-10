import re
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

IMG_DIR = Path("data/raw/images/US_Pictures")
OUT_DIR = Path("data/processed/images_roi")
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

def auto_crop_roi(img):
    """
    Ultrasound image-la usually black borders + scan info text irukkum edges-la.
    Idhai remove panni, central scan area mattum extract pannuvom.
    """
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    # Non-black (actual scan content) rows/columns kandupidikkum
    threshold = 15
    rows_with_content = np.where(gray.max(axis=1) > threshold)[0]
    cols_with_content = np.where(gray.max(axis=0) > threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return img  # fallback - crop panna mudiyala na original return pannum

    top, bottom = rows_with_content[0], rows_with_content[-1]
    left, right = cols_with_content[0], cols_with_content[-1]

    # Edges-la konjam margin vidrom (scan markers avoid panna)
    margin_h = int((bottom - top) * 0.05)
    margin_w = int((right - left) * 0.05)
    top = max(0, top + margin_h)
    bottom = min(h, bottom - margin_h)
    left = max(0, left + margin_w)
    right = min(w, right - margin_w)

    if bottom <= top or right <= left:
        return img

    cropped = img.crop((left, top, right, bottom))
    return cropped

IMG_SIZE = (224, 224)
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
            roi_img = auto_crop_roi(img)
            roi_img = roi_img.resize(IMG_SIZE)
            out_name = f"{pid}_{img_path.stem}.png"
            out_path = OUT_DIR / out_name
            roi_img.save(out_path)
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

print(f"\nTotal ROI-cropped images: {resized_count}")
print(f"Errors: {error_count}")

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv("data/processed/image_manifest_roi.csv", index=False)
print(f"Saved manifest: data/processed/image_manifest_roi.csv")
print(f"Manifest shape: {manifest_df.shape}")