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
- **PostgreSQL Database**: Stores user accounts and prediction history.
- **User Authentication**: JWT-based authentication with secure password hashing.
- **Admin Panel**: Separate admin interface for user management and prediction analytics.

## 🏃 Running Locally (Without Docker)

To run the application directly on your local machine without Docker:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Abasi-ifreke/medical-imaging.git
cd medical-imaging
```

### 2️⃣ Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
*Note: You may need to install `torch`, `tensorflow`, `fastapi`, `streamlit`, `uvicorn`, `pillow`, and other required libraries if requirements.txt is not provided.*

### 4️⃣ Train or download the model
- **Train your model:**  
  Run the following command if you want to train a new model:
  ```bash
  python app/train.py
  ```
  This will save a model file (e.g., `pneumonia_model.h5`) in the `app/` directory.


### 5️⃣ Start the FastAPI backend
```bash
python app/backend.py
```
The backend will be available at [http://localhost:8000](http://localhost:8000).

### 6️⃣ Start the Streamlit frontend
In a new terminal (with the virtual environment activated), run:
```bash
streamlit run app/frontend.py
```
The frontend will open at [http://localhost:8501](http://localhost:8501).

Now you can upload X-ray images and receive pneumonia predictions on your local machine!


## 🏃 Running With Docker
### 1️⃣ Prerequisites
Ensure you have the following installed:
- **Docker** & **Docker Compose**
- Python (for local testing)

### 2️⃣ Configure Environment Variables
Copy the example environment file and customize it:
```bash
cp .env.example .env
```
Edit `.env` to set your own values:
- `SECRET_KEY`: A secure random key for JWT tokens
- `ADMIN_EMAIL`: Default admin account email
- `ADMIN_PASSWORD`: Default admin account password

### 3️⃣ Build and Run the Containers
```bash
docker compose up --build
```
This will:
- Start a **PostgreSQL database** on port 5432
- Build the `med-app` (FastAPI backend) on **port 8000**
- Build the `med-frontend` (Streamlit frontend) on **port 8501**
- Build the `med-admin` (Admin panel) on **port 8502**

---
## 🔍 Usage

### User Interface
1. Open the **frontend** in your browser:
   ```
   http://localhost:8501
   ```
2. **Register** a new account or **login** with existing credentials.
3. Upload an X-ray image.
4. Click the **Diagnose** button.
5. The backend model will predict whether the image shows pneumonia or not.
6. View your **prediction history** in the sidebar.

### Admin Panel
1. Open the **admin panel** in your browser:
   ```
   http://localhost:8502
   ```
2. Login with admin credentials (default: `admin@example.com` / `admin123`).
3. View platform **statistics** and **analytics**.
4. **Manage users**: activate/deactivate accounts, change roles, delete users.
5. **Browse predictions**: filter by user, result type, and date.

---
## 🖼️ Example Results

### 1️⃣ Home Page
<img src="home.png" alt="Home Page" width="600"/>

### 2️⃣ Uploading a Chest X-ray
<img src="image upload.png" alt="Uploading Chest Xray" width="600"/>

### 3️⃣ Prediction Output
<img src="prediction.png" alt="Prediction Result" width="600"/>


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
│   ├── 📜 database.py            # Database configuration
│   ├── 📜 models.py              # SQLAlchemy models (User, Prediction)
│   ├── 📜 auth.py                # JWT authentication
│   ├── 📜 schemas.py             # Pydantic request/response schemas
│   ├── 📜 crud.py                # Database CRUD operations
│   ├── 📜 Dockerfile             # Backend Docker configuration
│   ├── 📜 requirements.txt       # Python dependencies
│   ├── 📂 alembic/               # Database migrations
│   └── 📦 pneumonia_model.h5     # Trained CNN model (generated)
│
├── 📂 frontend/                  # Frontend application
│   ├── 📜 frontend.py            # Streamlit web interface (with auth)
│   ├── 📜 admin.py               # Admin panel interface
│   ├── 📜 Dockerfile             # Frontend Docker configuration
│   ├── 📜 Dockerfile.admin       # Admin panel Docker configuration
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

### 2️⃣ Authentication Endpoints
**Register:** `POST /auth/register`
```python
import requests
response = requests.post("http://localhost:8000/auth/register", json={
    "email": "user@example.com",
    "password": "securepassword",
    "full_name": "John Doe"
})
```

**Login:** `POST /auth/login`
```python
response = requests.post("http://localhost:8000/auth/login", data={
    "username": "user@example.com",
    "password": "securepassword"
})
token = response.json()["access_token"]
```

### 3️⃣ Prediction Endpoint (Requires Authentication)
**Endpoint:** `POST /predict`

**Example Request:**
```python
import requests
headers = {"Authorization": f"Bearer {token}"}
files = {"file": ("image.png", open("xray.png", "rb"), "image/png")}
response = requests.post("http://localhost:8000/predict", files=files, headers=headers)
print(response.json())
```
**Response:**
```json
{
  "prediction": "Pneumonia",
  "confidence": 0.92,
  "message": "Pneumonia detected with 92.0% confidence.",
  "prediction_id": 1
}
```

### 4️⃣ User Endpoints
- `GET /auth/me` - Get current user profile
- `GET /predictions` - Get user's prediction history

### 5️⃣ Admin Endpoints (Admin Role Required)
- `GET /admin/users` - List all users
- `GET /admin/users/{id}` - Get specific user
- `PUT /admin/users/{id}` - Update user (activate/deactivate, change role)
- `DELETE /admin/users/{id}` - Delete user
- `GET /admin/predictions` - List all predictions (with filters)
- `GET /admin/stats` - Get platform statistics

---
## ☸️ Kubernetes Deployment

### 1️⃣ Create the namespace
```bash
kubectl create namespace medical-imaging
```

### 2️⃣ Apply the secrets (update values first!)
```bash
# Edit k8s/secrets.yaml with your production values
kubectl apply -f k8s/secrets.yaml
```

### 3️⃣ Deploy the database
```bash
kubectl apply -f k8s/database.yaml
```

### 4️⃣ Deploy the application
```bash
kubectl apply -f k8s/service-account.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 5️⃣ Verify deployment
```bash
kubectl get pods -n medical-imaging
kubectl get services -n medical-imaging
```
