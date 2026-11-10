
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import json

import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from catboost import CatBoostClassifier
from skimage import io as skio, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# ---------------- DIAGNOSIS MODEL ----------------

MODEL_PATH = OUT / "diagnosis_deployment.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Diagnosis model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

PREPROCESSOR = artifact["preprocessor"]
DIAGNOSIS_MODEL = artifact["model"]
SELECTED_MASK = artifact["selected_mask"]
INPUT_COLUMNS = artifact["input_columns"]
CLASS_NAMES = artifact["classes"]

print("Diagnosis model loaded successfully!")

# ---------------- PERITONITIS MODEL ----------------

PERITONITIS_PATH = OUT / "peritonitis_catboost_optuna.cbm"
DATA_PATH = ROOT / "data" / "processed" / "clinical_radiomics_fusion.csv"

if not PERITONITIS_PATH.exists():
    raise FileNotFoundError(f"Peritonitis model not found: {PERITONITIS_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

PERITONITIS_MODEL = CatBoostClassifier()
PERITONITIS_MODEL.load_model(str(PERITONITIS_PATH))

PERITONITIS_FEATURES = PERITONITIS_MODEL.feature_names_

training_df = pd.read_csv(DATA_PATH)

missing_features = [
    c for c in PERITONITIS_FEATURES
    if c not in training_df.columns
]

if missing_features:
    raise ValueError(
        f"Training dataset is missing model features: {missing_features}"
    )

if "Peritonitis" not in training_df.columns:
    raise ValueError("Peritonitis target column is missing.")

d = training_df.loc[training_df["Peritonitis"].notna()].copy()

X = d[PERITONITIS_FEATURES].copy()
y = d["Peritonitis"].astype(str)

for column in X.columns:
    if pd.api.types.is_numeric_dtype(X[column]):
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    else:
        X[column] = X[column].fillna("__missing__").astype(str)

PERITONITIS_CAT_COLUMNS = [
    c for c in X.columns
    if not pd.api.types.is_numeric_dtype(X[c])
]

PERITONITIS_NUM_COLUMNS = [
    c for c in PERITONITIS_FEATURES
    if c not in PERITONITIS_CAT_COLUMNS
]

X_train, X_hold, y_train, y_hold = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

PERITONITIS_MEDIANS = (
    X_train[PERITONITIS_NUM_COLUMNS].median().fillna(0)
)

print("Peritonitis model loaded successfully!")
print("Peritonitis features:", len(PERITONITIS_FEATURES))
print("Peritonitis classes:", list(PERITONITIS_MODEL.classes_))


# ---------------- FASTAPI APPLICATION ----------------

app = FastAPI(
    title="Pediatric Appendicitis and Peritonitis Diagnosis",
    description="Multimodal, Feature-Optimized and Uncertainty-Aware Framework",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PatientData(BaseModel):
    age: int = Field(ge=0, le=18)
    bmi: float = Field(gt=0, le=100)
    wbc_count: float = Field(ge=0)
    appendix_diameter: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Pediatric AI Diagnosis API is running",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "project": "Pediatric Appendicitis FYP",
        "diagnosis_model": "loaded",
        "peritonitis_model": "loaded",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/patient/validate")
def validate_patient(patient: PatientData):
    return {
        "status": "validated",
        "message": "Clinical input received successfully",
        "patient_data": patient.model_dump(),
        "prediction_available": True
    }


# ---------------- DIAGNOSIS PREDICTION ----------------

@app.post("/predict/diagnosis")
def predict_diagnosis(payload: dict):
    try:
        patient_row = pd.DataFrame(
            [{
                col: payload.get(col, np.nan)
                for col in INPUT_COLUMNS
            }],
            columns=INPUT_COLUMNS
        )

        processed = PREPROCESSOR.transform(patient_row)
        selected_data = processed[:, SELECTED_MASK]

        prediction = DIAGNOSIS_MODEL.predict(selected_data)[0]
        probabilities = DIAGNOSIS_MODEL.predict_proba(selected_data)[0]

        return {
            "status": "success",
            "prediction": str(CLASS_NAMES[int(prediction)]),
            "probabilities": {
                str(CLASS_NAMES[i]): round(float(probabilities[i]), 4)
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- ULTRASOUND VALIDATION ----------------

@app.post("/ultrasound/validate")
async def validate_ultrasound(file: UploadFile = File(...)):

    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are allowed"
        )

    content = await file.read(5 * 1024 * 1024 + 1)

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MB limit"
        )

    if not content:
        raise HTTPException(status_code=400, detail="Image is empty")

    try:
        with Image.open(BytesIO(content)) as img:
            image_format = img.format
            width, height = img.size
            img.verify()

        if image_format not in {"PNG", "JPEG"}:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    return {
        "status": "validated",
        "filename": file.filename,
        "size_bytes": len(content),
        "image_width": width,
        "image_height": height,
        "message": "Ultrasound image validated successfully"
    }


# ---------------- RADIOMICS EXTRACTION ----------------

def extract_uploaded_radiomics(content: bytes):

    img = skio.imread(BytesIO(content))

    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])

    img = img_as_ubyte(img)

    px = img.ravel().astype(float)

    features = {
        "mean": float(px.mean()),
        "std": float(px.std()),
        "skew": float(stats.skew(px)),
        "kurtosis": float(stats.kurtosis(px)),
        "min": float(px.min()),
        "max": float(px.max()),
        "median": float(np.median(px)),
        "p10": float(np.percentile(px, 10)),
        "p90": float(np.percentile(px, 90)),
        "entropy": float(
            stats.entropy(
                np.bincount(img.ravel(), minlength=256) + 1e-9
            )
        )
    }

    quantized = (img // 8).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1, 3],
        angles=[0, np.pi / 2],
        levels=32,
        symmetric=True,
        normed=True
    )

    for prop in [
        "contrast",
        "homogeneity",
        "energy",
        "correlation",
        "dissimilarity"
    ]:
        features["glcm_" + prop] = float(
            graycoprops(glcm, prop).mean()
        )

    return features


# ---------------- PERITONITIS PREDICTION ----------------

@app.post("/predict/peritonitis")
async def predict_peritonitis(
    payload: str = Form(...),
    file: UploadFile = File(...)
):

    try:
        try:
            patient_data = json.loads(payload)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Clinical payload must contain valid JSON"
            )

        if not isinstance(patient_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Clinical payload must be a JSON object"
            )

        if file.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(
                status_code=400,
                detail="Only PNG and JPEG images are allowed"
            )

        content = await file.read(5 * 1024 * 1024 + 1)

        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 5 MB limit"
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        try:
            with Image.open(BytesIO(content)) as img:
                actual_format = img.format
                img.verify()

            if actual_format not in {"PNG", "JPEG"}:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format"
                )

        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image"
            )

        radiomics = extract_uploaded_radiomics(content)

        row_data = {}

        for column in PERITONITIS_FEATURES:
            if column in radiomics:
                row_data[column] = radiomics[column]
            else:
                row_data[column] = patient_data.get(column, np.nan)

        row = pd.DataFrame(
            [row_data],
            columns=PERITONITIS_FEATURES
        )

        for column in PERITONITIS_NUM_COLUMNS:
            row[column] = pd.to_numeric(
                row[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            row[column] = row[column].fillna(
                PERITONITIS_MEDIANS[column]
            )

        for column in PERITONITIS_CAT_COLUMNS:
            row[column] = row[column].fillna(
                "__missing__"
            ).astype(str)

        prediction = str(
            np.asarray(PERITONITIS_MODEL.predict(row)).ravel()[0]
        )

        probabilities = PERITONITIS_MODEL.predict_proba(row)[0]

        return {
            "status": "success",
            "prediction": prediction,
            "probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    PERITONITIS_MODEL.classes_,
                    probabilities
                )
            },
            "message": "Peritonitis prediction generated successfully",
            "note": (
                "Experimental single-image inference. "
                "Not for clinical decision-making."
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- START SERVER ----------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
