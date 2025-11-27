import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load all models
xray_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_model.load_state_dict(torch.load("xray_model.pth", map_location=torch.device('cpu')))
xray_model.eval()

lung_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
lung_model.fc = nn.Linear(lung_model.fc.in_features, 2)
lung_model.load_state_dict(torch.load("lung_model.pth", map_location=torch.device('cpu')))
lung_model.eval()

pneumonia_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
pneumonia_model.fc = nn.Linear(pneumonia_model.fc.in_features, 2)
pneumonia_model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
pneumonia_model.eval()

# FastAPI setup
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    # Confidence thresholds
    XRAY_CONFIDENCE_THRESHOLD = 0.7
    LUNG_CONFIDENCE_THRESHOLD = 0.7
    PNEUMONIA_CONFIDENCE_THRESHOLD = 0.6

    result = {
        "is_xray": False,
        "is_lung_xray": False,
        "prediction": None,
        "confidence": None,
        "message": None
    }

    with torch.no_grad():
        # Check if image is an X-ray
        xray_output = xray_model(tensor)
        xray_probs = torch.softmax(xray_output, dim=1)
        xray_confidence = xray_probs[0][1].item()
        
        if xray_confidence < XRAY_CONFIDENCE_THRESHOLD:
            result["message"] = "This does not appear to be an X-ray image. Please upload a chest X-ray."
            result["confidence"] = xray_confidence
            return result
        
        result["is_xray"] = True
        
        # Check if it's a lung X-ray
        lung_output = lung_model(tensor)
        lung_probs = torch.softmax(lung_output, dim=1)
        lung_confidence = lung_probs[0][1].item()
        
        if lung_confidence < LUNG_CONFIDENCE_THRESHOLD:
            result["message"] = "This appears to be an X-ray, but not a chest/lung X-ray."
            result["confidence"] = lung_confidence
            return result
        
        result["is_lung_xray"] = True
        
        # Classify pneumonia
        pneumonia_output = pneumonia_model(tensor)
        pneumonia_probs = torch.softmax(pneumonia_output, dim=1)
        pneumonia_confidence = pneumonia_probs[0][1].item()
        normal_confidence = pneumonia_probs[0][0].item()
        
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