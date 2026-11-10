import re
from pathlib import Path
import pandas as pd

IMG = Path("data/raw/images/US_Pictures")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")

pattern = re.compile(r"^(\d+)\.(\d+)")
patient_imgs = {}
for p in IMG.iterdir():
    if p.is_file():
        m = pattern.match(p.name)
        if m:
            pid = int(m.group(1))
            patient_imgs.setdefault(pid, []).append(str(p.relative_to(Path("data/raw"))))

df["US_Number_int"] = df["US_Number"].dropna().astype(int)
df["image_count"] = df["US_Number_int"].map(lambda x: len(patient_imgs.get(x, [])) if pd.notna(x) else 0)
df["has_images"] = df["image_count"] > 0

print("Total rows:", len(df))
print("Rows with at least 1 matched image:", df["has_images"].sum())
print("Rows with no matched image:", (~df["has_images"]).sum())

multimodal_ids = df[df["has_images"]].index
print("\nMultimodal-ready patients:", len(multimodal_ids))

save_df = df.copy()
save_df.to_csv(OUT / "patients_with_image_flag.csv", index=False)
print("\nSaved: data/processed/patients_with_image_flag.csv")

print("\nDiagnosis balance in multimodal subset:")
print(df.loc[multimodal_ids, "Diagnosis"].value_counts(dropna=False))
print("\nSeverity balance in multimodal subset:")
print(df.loc[multimodal_ids, "Severity"].value_counts(dropna=False))