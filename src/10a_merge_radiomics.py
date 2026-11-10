import pandas as pd
from pathlib import Path

df_clinical = pd.read_excel("data/raw/tabular/app_data.xlsx", sheet_name="All cases")
df_clinical["US_Number_int"] = df_clinical["US_Number"].dropna().astype(int)

df_radiomics = pd.read_csv("data/processed/roi_features.csv")

merged = df_clinical.merge(
    df_radiomics, left_on="US_Number_int", right_on="patient_id", how="inner"
)

OUT = Path("data/processed/clinical_radiomics_fusion.csv")
merged.to_csv(OUT, index=False)
print("Merged shape:", merged.shape)
print("Saved:", OUT)