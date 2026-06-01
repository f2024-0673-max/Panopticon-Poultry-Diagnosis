from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Panopticon Poultry Disease Detection API",
    description="MobileNetV2-based classifier for broiler fecal disease detection",
    version="1.0.0"
)

# CORS — allows Streamlit or any frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = "model.keras"
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/")
def root():
    return {"status": "Panopticon API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images accepted")

    image_bytes = await file.read()

    try:
        arr = preprocess(image_bytes)
        preds = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    probabilities = {
        CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[np.argmax(preds)],
        "confidence": round(float(np.max(preds)) * 100, 2),
        "probabilities": probabilities
    }
