# Medical Imaging Application - Pneumonia Detection

This project is a **medical diagnosis web application** that uses Convolutional Neural Network (CNN) to detect **pneumonia from X-ray images**. It consists of a **FastAPI backend** for model inference and a **Streamlit frontend** for user interaction.

## 🎯 Problem Statement

Following a chest X-ray examination, medical technicians successfully acquire diagnostic images. However, radiologists or healthcare professionals may not be immediately available to interpret the scans and provide a timely diagnosis. This delay can impact patient care, particularly in resource-constrained settings or emergency situations.

**Research Question:** Can a Convolutional Neural Network (CNN) model serve as an automated first-line screening tool to detect pneumonia from chest X-ray scans, providing rapid preliminary assessments while maintaining clinical accuracy?

## 🔬 Solution Approach

This project implements a deep learning-based classification system that:
- Analyzes chest X-ray images using custom CNN architecture trained from scratch
- Provides binary classification (Normal vs. Pneumonia) with confidence scores
- Employs automated hyperparameter tuning for optimal model performance
- Maintains transparency by displaying confidence levels and recommending professional consultation for low-confidence predictions

## 🚀 Features
- **FastAPI Backend**: Handles image processing and model predictions.
- **Streamlit Frontend**: Provides an easy-to-use interface for image uploads.
- **Dockerized Deployment**: Uses Docker Compose for easy container orchestration.
- **Tensorflow**: Powers the deep learning pipeline, including model building, training, evaluation, and inference for detecting pneumonia from chest X-ray images with a custom Convolutional Neural Network.

---
## 📦 Setup & Installation
### 1️⃣ Prerequisites
Ensure you have the following installed:
- **Docker** & **Docker Compose**
- Python (for local testing)

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/Abasi-ifreke/medical-imaging.git
cd medical-imaging
```

### 3️⃣ Build and Run the Containers
```bash
docker compose up --build
```
This will:
- Build the `med-app` (FastAPI backend) and `med-frontend` (Streamlit frontend) containers.
- Expose the backend on **port 8000** and frontend on **port 8501**.

---
## 🔍 Usage
1. Open the **frontend** in your browser:
   ```
   http://localhost:8501
   ```
2. Upload an X-ray image.
3. Click the **Diagnose** button.
4. The backend model will predict whether the image shows pneumonia or not.

---
## ⚙️ Project Structure
```
📂 medical-imaging/
│
├── 📜 docker-compose.yaml        # Docker Compose orchestration
├── 📜 README.md                  # Project documentation
├── 📜 .gitignore                 # Git ignore rules
├── 📓 notebook.ipynb             # Jupyter notebook for training & experimentation
│
├── 📂 app/                       # Backend application
│   ├── 📜 backend.py             # FastAPI inference server
│   ├── 📜 train.py               # Model training script
│   ├── 📜 Dockerfile             # Backend Docker configuration
│   ├── 📜 requirements.txt       # Python dependencies
│   └── 📦 pneumonia_model.h5     # Trained CNN model (generated)
│
├── 📂 frontend/                  # Frontend application
│   ├── 📜 frontend.py            # Streamlit web interface
│   ├── 📜 Dockerfile             # Frontend Docker configuration
│   └── 📜 requirements.txt       # Frontend dependencies
│
└── 📂 data/                      # Dataset (not in git)
    ├── 📂 train/                 # Training images
    │   ├── 📂 NORMAL/            # Normal chest X-rays
    │   └── 📂 PNEUMONIA/         # Pneumonia chest X-rays
    ├── 📂 val/                   # Validation images
    │   ├── 📂 NORMAL/
    │   └── 📂 PNEUMONIA/
    └── 📂 test/                  # Test images
        ├── 📂 NORMAL/
        └── 📂 PNEUMONIA/
```

---
## 🛠 API Endpoints
### 1️⃣ Test API (Swagger UI)
Once running, access the API docs at:
```
http://localhost:8000/docs
```

### 2️⃣ Prediction Endpoint
**Endpoint:** `POST /predict`

**Example Request:**
```python
import requests
files = {"file": ("image.png", open("xray.png", "rb"), "image/png")}
response = requests.post("http://localhost:8000/predict", files=files)
print(response.json())
```
**Response:**
```json
{
  "prediction": "Pneumonia"
}
```
