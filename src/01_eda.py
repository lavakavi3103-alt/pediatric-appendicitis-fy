from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("data/raw")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. Tabular file kandupidikkum (xlsx / csv)
tab_files = [p for p in RAW.rglob("*") if p.suffix.lower() in (".xlsx", ".csv")]
print("Tabular files found:")
for p in tab_files:
    print("  ", p)
if not tab_files:
    raise SystemExit("Tabular file kedaikkala. data/raw kulla copy pannirukka nu check pannunga.")

# xlsx irundha adhu first, illana csv
tab_files.sort(key=lambda p: (p.suffix.lower() == ".csv", str(p)))
path = tab_files[0]
print("\nUsing:", path)

if path.suffix.lower() == ".csv":
    df = pd.read_csv(path)
else:
    print("Sheets:", pd.ExcelFile(path).sheet_names)
    df = pd.read_excel(path)

# 2. Images count
img_ext = {".bmp", ".png", ".jpg", ".jpeg"}
imgs = [p for p in RAW.rglob("*") if p.suffix.lower() in img_ext]
print("\nImages found:", len(imgs))
if imgs:
    print("Sample names:", [p.name for p in imgs[:5]])

# 3. Basic info
print("\nShape (rows, columns):", df.shape)
print("\nDtypes:\n", df.dtypes.value_counts())

with open(OUT / "columns.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, df.columns)))
print("\nAll column names outputs/columns.txt la save aayiduchu")

# 4. Target columns
for col in df.columns:
    if any(k in str(col).lower() for k in ["diagnosis", "severity", "management"]):
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))

# 5. Patient ID column irukka
for col in df.columns:
    if "id" in str(col).lower() or "subject" in str(col).lower():
        print(f"\nID-like column: {col}, unique values: {df[col].nunique()}")

# 6. Missing values
missing = df.isnull().mean().sort_values(ascending=False)
print("\nTop 15 missing columns:\n", missing.head(15))

if (missing > 0).any():
    missing[missing > 0].plot(kind="barh", figsize=(8, 10))
    plt.title("Missing value ratio")
    plt.tight_layout()
    plt.savefig(OUT / "missing_values.png")
    print("\nPlot outputs/missing_values.png la save aayiduchu")
else:
    print("\nMissing values illa")