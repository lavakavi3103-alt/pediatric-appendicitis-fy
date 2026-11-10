import re
from pathlib import Path
from collections import Counter
import pandas as pd

IMG = Path("data/raw/images/US_Pictures")
df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")

files = [p for p in IMG.iterdir() if p.is_file()]
# Filename starts with <id>.<view>, edhu irundhaalum appuram (text/ext) ignore pannuvom
pattern = re.compile(r"^(\d+)\.(\d+)")

patient_imgs, other = {}, []
for p in files:
    m = pattern.match(p.name)
    if m:
        patient_imgs.setdefault(int(m.group(1)), []).append(p.name)
    else:
        other.append(p.name)

print("Total files:", len(files))
print("Files matching '<id>.<view>...' at start:", sum(len(v) for v in patient_imgs.values()))
print("Unique patient IDs found in images:", len(patient_imgs))
print("Other files (no leading number):", len(other))
print("Other sample:", sorted(other)[:10])

counts = [len(v) for v in patient_imgs.values()]
if counts:
    print("\nImages per patient: min", min(counts), "max", max(counts))
    dist = Counter(counts)
    print("Distribution (views:num_patients), first 10:", dict(sorted(dist.items())[:10]))
    first = sorted(patient_imgs)[:3]
    for k in first:
        print(f"Sample patient {k}:", sorted(patient_imgs[k]))

# Tabular side
us = df["US_Number"].dropna().astype(int)
print("\nRows with US_Number:", len(us), "| unique:", us.nunique())
print("US_Number min/max:", us.min(), us.max())

tab_ids, img_ids = set(us), set(patient_imgs)
inter = tab_ids & img_ids
print("\nIDs in BOTH table (US_Number) and images:", len(inter))
print("In table but no images:", len(tab_ids - img_ids))
print("In images but not in table:", len(img_ids - tab_ids), sorted(img_ids - tab_ids)[:10])

# Onnum match aagalana, index/row number-a try pannuvom
row_ids = set(range(1, len(df) + 1))
print("\n--- Alternative: image ID vs row number (1..782) ---")
print("IDs in both row-number and images:", len(row_ids & img_ids))