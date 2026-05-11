from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


# --- Auth Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None


# --- Prediction Schemas ---

class PredictionResponse(BaseModel):
    id: int
    image_filename: str
    result: str
    confidence: float
    message: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionWithUser(PredictionResponse):
    user_id: int
    user_email: Optional[str] = None


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    message: str
    prediction_id: Optional[int] = None


# --- Admin Schemas ---

class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int


class PredictionListResponse(BaseModel):
    predictions: List[PredictionWithUser]
    total: int


class StatsResponse(BaseModel):
    total_users: int
    active_users: int
    total_predictions: int
    pneumonia_count: int
    normal_count: int
    predictions_today: int
