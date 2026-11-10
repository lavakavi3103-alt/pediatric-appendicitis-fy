
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DATA = ROOT / "data/processed/clinical_radiomics_fusion.csv"
FEATURE_FILE = OUT / "dsa_selected_features.txt"

df = pd.read_csv(DATA)

with open(FEATURE_FILE, encoding="utf-8") as f:
    features = [
        line.strip() for line in f if line.strip()
    ]

# Remove post-outcome and potentially leaking features
exclude = {
    "Length_of_Stay",
    "Management",
    "Diagnosis",
    "Severity",
    "Peritonitis",
    "Diagnosis_Presumptive",
    "Perforation",
    "Appendicular_Abscess",
}

features = [
    c for c in features
    if c in df.columns and c not in exclude
]

if not features:
    raise ValueError("No valid clinical features found.")

df = df[df["Diagnosis"].notna()].copy()

X = df[features].copy()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(
    df["Diagnosis"].astype(str)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_cols = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_cols = [
    c for c in features if c not in numeric_cols
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="missing"
    )),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    verbose_feature_names_out=False
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

pred = model.predict(X_test_processed)

print("=== SHAP EXPLAINABILITY ===")
print("Classes:", list(label_encoder.classes_))
print("Clinical features:", len(features))
print("Train/Test:", len(X_train), len(X_test))
print(
    "Test accuracy:",
    round(accuracy_score(y_test, pred) * 100, 2),
    "%"
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(
    X_test_processed
)

if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

# SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_processed,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUT / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

# SHAP importance bar chart
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_processed,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    OUT / "shap_importance_bar.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close("all")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.abs(
        shap_values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    OUT / "shap_feature_importance.csv",
    index=False
)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

print("\nSaved:")
print("outputs/shap_summary.png")
print("outputs/shap_importance_bar.png")
print("outputs/shap_feature_importance.csv")
print("\nSHAP completed successfully.")
print(
    "NOTE: Earlier DSA feature selection may "
    "have full-dataset bias."
)
