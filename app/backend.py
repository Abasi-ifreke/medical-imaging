import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

# -----------------------------
# Load Pneumonia model
# -----------------------------
pneumonia_model = tf.keras.models.load_model("pneumonia_model.h5")

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
    probs = pneumonia_model.predict(tensor)[0]
    pneumonia_confidence = float(probs[1])  # Assuming [Normal, Pneumonia]
    normal_confidence = float(probs[0])

    if pneumonia_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD and normal_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD:
        result["message"] = "Unable to make a confident prediction. Please upload a clearer chest X-ray image."
        result["confidence"] = max(pneumonia_confidence, normal_confidence)
        return result

    if pneumonia_confidence > normal_confidence:
        result["prediction"] = "Pneumonia"
        result["confidence"] = pneumonia_confidence
        result["message"] = f"Pneumonia detected with {pneumonia_confidence*100:.1f}% confidence."
    else:
        result["prediction"] = "Normal"
        result["confidence"] = normal_confidence
        result["message"] = f"No pneumonia detected. Lungs appear normal with {normal_confidence*100:.1f}% confidence."

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)