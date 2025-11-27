# Medical Imaging Application - Pneumonia Detection

This project is a **medical diagnosis web application** that detects **pneumonia from X-ray images** using a deep learning model. It consists of a **FastAPI backend** for model inference and a **Streamlit frontend** for user interaction.

## 🚀 Features
- **FastAPI Backend**: Handles image processing and model predictions.
- **Streamlit Frontend**: Provides an easy-to-use interface for image uploads.
- **Dockerized Deployment**: Uses Docker Compose for easy container orchestration.

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
│── 📜 docker-compose.yml    # Docker Compose configuration
│── 📂 app/                  
    │── backend.py            # FastAPI backend
    │── Dockerfile            # Dockefile for image build
    │── pneumonia_model.pth   # Trained model
    │── requirements.txt      # Application requirement
    └── train.py              # Script to build a trained model                
└── 📂 frontend/                  
    │── frontend.py           # Streamlit UI
    │── Dockerfile            # Dockefile for image build
    └── requirements.txt      # Python dependencies
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
