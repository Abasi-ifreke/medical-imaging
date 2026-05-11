import os
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordRequestForm
from PIL import Image
import numpy as np
import io
from typing import Optional
from sqlalchemy.orm import Session

from database import get_db, init_db
from models import User, Prediction
from auth import (
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin,
)
from schemas import (
    Token,
    UserCreate,
    UserResponse,
    UserUpdate,
    UserListResponse,
    PredictionResponse,
    PredictionListResponse,
    PredictionResult,
    PredictionWithUser,
    StatsResponse,
)
import crud

MODEL_PATH = os.getenv("MODEL_PATH", "/pneumonia_model.h5")

print(f"Loading model from: {MODEL_PATH}")

pneumonia_model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="Medical Imaging API", version="2.0.0")


@app.on_event("startup")
def on_startup():
    """Initialize database tables on startup."""
    init_db()
    
    db = next(get_db())
    try:
        admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        
        existing_admin = crud.get_user_by_email(db, admin_email)
        if not existing_admin:
            crud.create_user(
                db,
                email=admin_email,
                password=admin_password,
                full_name="Administrator",
                role="admin"
            )
            print(f"Default admin user created: {admin_email}")
    finally:
        db.close()


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize and normalize image for TensorFlow model."""
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Health Check ---

@app.get("/health")
async def get_health():
    return {"status": "healthy"}


# --- Authentication Endpoints ---

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    existing_user = crud.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = crud.create_user(
        db,
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name
    )
    return user


@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and receive an access token."""
    user = crud.get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get the current user's profile."""
    return current_user


# --- Prediction Endpoint (Protected) ---

@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload an X-ray image for pneumonia prediction."""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess_image(image)

    PNEUMONIA_CONFIDENCE_THRESHOLD = 0.6

    prediction_value = pneumonia_model.predict(tensor)[0][0]
    pneumonia_confidence = float(prediction_value)
    normal_confidence = float(1 - prediction_value)

    if pneumonia_confidence > normal_confidence:
        result = "Pneumonia"
        confidence = pneumonia_confidence
        if pneumonia_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD:
            message = f"Pneumonia detected with {pneumonia_confidence*100:.1f}% confidence. Low confidence - please consult a medical professional for confirmation."
        else:
            message = f"Pneumonia detected with {pneumonia_confidence*100:.1f}% confidence."
    else:
        result = "Normal"
        confidence = normal_confidence
        if normal_confidence < PNEUMONIA_CONFIDENCE_THRESHOLD:
            message = f"No pneumonia detected with {normal_confidence*100:.1f}% confidence. Low confidence - please consult a medical professional for confirmation."
        else:
            message = f"No pneumonia detected. Lungs appear normal with {normal_confidence*100:.1f}% confidence."

    prediction_record = crud.create_prediction(
        db,
        user_id=current_user.id,
        image_filename=file.filename or "uploaded_image.png",
        result=result,
        confidence=confidence,
        message=message
    )

    return {
        "prediction": result,
        "confidence": confidence,
        "message": message,
        "prediction_id": prediction_record.id
    }


# --- User Prediction History ---

@app.get("/predictions", response_model=PredictionListResponse)
async def get_my_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the current user's prediction history."""
    predictions, total = crud.get_user_predictions(db, current_user.id, skip, limit)
    
    prediction_list = []
    for p in predictions:
        prediction_list.append(PredictionWithUser(
            id=p.id,
            image_filename=p.image_filename,
            result=p.result,
            confidence=p.confidence,
            message=p.message,
            created_at=p.created_at,
            user_id=p.user_id,
            user_email=current_user.email
        ))
    
    return {"predictions": prediction_list, "total": total}


# --- Admin Endpoints ---

@app.get("/admin/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users, total = crud.get_users(db, skip, limit)
    return {"users": users, "total": total}


@app.get("/admin/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get a specific user by ID (admin only)."""
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@app.put("/admin/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Update a user's information (admin only)."""
    user = crud.update_user(
        db,
        user_id,
        full_name=user_update.full_name,
        is_active=user_update.is_active,
        role=user_update.role
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Delete a user (admin only)."""
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    success = crud.delete_user(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"message": "User deleted successfully"}


@app.get("/admin/predictions", response_model=PredictionListResponse)
async def list_all_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = Query(None),
    result_filter: Optional[str] = Query(None),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all predictions with optional filters (admin only)."""
    predictions, total = crud.get_all_predictions(db, skip, limit, user_id, result_filter)
    
    prediction_list = []
    for p in predictions:
        user = crud.get_user_by_id(db, p.user_id)
        prediction_list.append(PredictionWithUser(
            id=p.id,
            image_filename=p.image_filename,
            result=p.result,
            confidence=p.confidence,
            message=p.message,
            created_at=p.created_at,
            user_id=p.user_id,
            user_email=user.email if user else None
        ))
    
    return {"predictions": prediction_list, "total": total}


@app.get("/admin/stats", response_model=StatsResponse)
async def get_stats(
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get platform statistics (admin only)."""
    return crud.get_stats(db)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
