"""
api.py
------
FastAPI REST backend for the Fruit Quality Analyzer.

Endpoints:
    POST /predict          Upload an image → get quality + shelf-life prediction
    GET  /history          Retrieve past prediction records
    GET  /stats            Aggregate statistics
    GET  /health           Health check
    DELETE /history        Clear all history

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import predict_from_array, load_model_and_meta
from database import fetch_history, fetch_stats, clear_history
from utils.preprocess import IMG_SIZE

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="🍎 Fruit Quality Analyzer API",
    description=(
        "AI-powered fruit & vegetable quality classification "
        "and shelf-life estimation using deep learning (MobileNetV2 / ResNet50)."
    ),
    version="1.0.0",
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────

class ShelfLifeInfo(BaseModel):
    days:   str
    emoji:  str
    status: str


class Top3Item(BaseModel):
    cls:        str
    confidence: float

    class Config:
        fields = {"cls": "class"}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence:      float
    shelf_life:      ShelfLifeInfo
    top3:            list
    all_probs:       dict
    image_id:        str


class HistoryRecord(BaseModel):
    id:              int
    timestamp:       str
    filename:        Optional[str]
    predicted_class: str
    confidence:      float
    shelf_life_days: Optional[str]
    shelf_status:    Optional[str]
    top3:            list


class StatsResponse(BaseModel):
    total:          int
    by_class:       dict
    avg_confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# Startup: warm up model
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load the model into memory once at server start."""
    try:
        load_model_and_meta()
        log.info("Model loaded successfully ✓")
    except FileNotFoundError as e:
        log.warning(f"Model not found at startup: {e}")
        log.warning("Run `python train.py` first, then restart the API.")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Simple liveness check."""
    return {"status": "ok", "service": "Fruit Quality Analyzer API"}


@app.get("/classes", tags=["Model"])
async def get_classes():
    """Return the list of class names the model was trained on."""
    try:
        _, class_names = load_model_and_meta()
        return {"class_names": class_names, "count": len(class_names)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image file (JPEG/PNG)")):
    """
    Upload a fruit/vegetable image and receive:
    - Quality classification (e.g. fresh / medium / rotten)
    - Confidence score
    - Shelf-life estimate
    - Top-3 alternative predictions
    """
    # ── Validate file type ────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG."
        )

    # ── Read & decode image ───────────────────────────────────────────────
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Save upload ───────────────────────────────────────────────────────
    image_id   = uuid.uuid4().hex[:12]
    save_path  = UPLOAD_DIR / f"{image_id}_{file.filename}"
    cv2.imwrite(str(save_path), img_bgr)

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        result = predict_from_array(img_rgb, filename=file.filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        predicted_class = result["predicted_class"],
        confidence      = result["confidence"],
        shelf_life      = ShelfLifeInfo(**result["shelf_life"]),
        top3            = result["top3"],
        all_probs       = result["all_probs"],
        image_id        = image_id,
    )


@app.get("/history", tags=["Database"])
async def get_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieve the most recent `limit` prediction records."""
    records = fetch_history(limit=limit)
    return {"count": len(records), "records": records}


@app.get("/stats", response_model=StatsResponse, tags=["Database"])
async def get_stats():
    """Return aggregate statistics across all stored predictions."""
    return fetch_stats()


@app.delete("/history", tags=["Database"])
async def delete_history():
    """Clear all prediction history from the database."""
    clear_history()
    return {"message": "History cleared."}


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
