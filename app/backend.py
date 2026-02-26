import os
import tensorflow as tf
from google.cloud import storage
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

BUCKET_NAME = os.getenv("MODEL_BUCKET")
MODEL_BLOB_NAME = os.getenv("MODEL_NAME")
LOCAL_MODEL_PATH = "/pneumonia_model.h5"

# -----------------------------
# Download model from GCS (if not already downloaded)
# -----------------------------
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Downloading model from GCS...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB_NAME)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")
    else:
        print("Model already exists locally.")

download_model()

# -----------------------------
# Load Pneumonia model
# -----------------------------
pneumonia_model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize and normalize image for TensorFlow model."""
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess_image(image)

    # Confidence threshold
    PNEUMONIA_CONFIDENCE_THRESHOLD = 0.6

    result = {
        "prediction": None,
        "confidence": None,
        "message": None
    }

    # -----------------------------
    # Classify pneumonia
    # -----------------------------
    # Model outputs single value: 0 (Normal) to 1 (Pneumonia)
    prediction = pneumonia_model.predict(tensor)[0][0]
    pneumonia_confidence = float(prediction)
    normal_confidence = float(1 - prediction)

    # Determine prediction
    if pneumonia_confidence > normal_confidence:
        result["prediction"] = "Pneumonia"
        result["confidence"] = pneumonia_confidence
        
        # Add disclaimer if confidence is low
        if pneumonia_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD:
            result["message"] = f"Pneumonia detected with {pneumonia_confidence*100:.1f}% confidence. ⚠️ Low confidence - please consult a medical professional for confirmation."
        else:
            result["message"] = f"Pneumonia detected with {pneumonia_confidence*100:.1f}% confidence."
    else:
        result["prediction"] = "Normal"
        result["confidence"] = normal_confidence
        
        # Add disclaimer if confidence is low
        if normal_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD:
            result["message"] = f"No pneumonia detected with {normal_confidence*100:.1f}% confidence. ⚠️ Low confidence - please consult a medical professional for confirmation."
        else:
            result["message"] = f"No pneumonia detected. Lungs appear normal with {normal_confidence*100:.1f}% confidence."

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)