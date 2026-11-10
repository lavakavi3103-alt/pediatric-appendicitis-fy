from pathlib import Path
from collections import Counter
import pandas as pd

IMG = Path("data/raw/images")
if not IMG.exists():
    raise SystemExit("data/raw/images folder kedaikkala")

files = [p for p in IMG.rglob("*") if p.is_file()]
print("Total files:", len(files))

# 1. File types
exts = Counter(p.suffix.lower() for p in files)
print("\nFile types:", dict(exts))

# 2. Folders and file counts
folders = Counter(p.parent.relative_to(IMG) for p in files)
print("\nFolders (file count):")
for f, n in sorted(folders.items(), key=lambda x: str(x[0])):
    print(f"  {f}: {n}")

# 3. Sample names per folder
print("\nSample names per folder:")
for f in sorted(folders, key=str):
    names = sorted(p.name for p in (IMG / f).iterdir() if p.is_file())
    print(f"  [{f}] ->", names[:6])

# 4. Tabular-la patient/image link column irukka
df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
keys = ["number", "patient", "_id", "us_"]
link_cols = [c for c in df.columns if any(k in str(c).lower() for k in keys)]
print("\nPossible link columns:", link_cols)
for c in link_cols:
    print(f"\n{c} (first 10):", df[c].head(10).tolist())