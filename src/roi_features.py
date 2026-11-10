import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
import re

IMG_DIR = Path("data/raw/images")      # ungal ultrasound images folder
OUT = Path("data/processed/roi_features.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def extract_features(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    img = img_as_ubyte(img)
    px = img.ravel().astype(float)

    f = {
        "mean": px.mean(), "std": px.std(),
        "skew": stats.skew(px), "kurtosis": stats.kurtosis(px),
        "min": px.min(), "max": px.max(),
        "median": np.median(px),
        "p10": np.percentile(px, 10), "p90": np.percentile(px, 90),
        "entropy": stats.entropy(np.bincount(img.ravel(), minlength=256) + 1e-9),
    }

    q = (img // 8).astype(np.uint8)  # 32 gray levels
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi/2],
                        levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        f["glcm_" + prop] = graycoprops(glcm, prop).mean()
    return f

rows = []
files = [p for p in IMG_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}]
print("Images found:", len(files))

for p in files:
    m = re.match(r"(\d+)", p.stem)   # filename modhalla irukkura number = patient id
    if not m:
        continue
    try:
        feats = extract_features(p)
        feats["patient_id"] = int(m.group(1))
        rows.append(feats)
    except Exception as e:
        print("Skipped", p.name, e)

df = pd.DataFrame(rows).groupby("patient_id", as_index=False).mean()
df.to_csv(OUT, index=False)
print("Saved:", OUT, "| shape:", df.shape)