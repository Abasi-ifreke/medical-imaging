from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import User, Prediction
from auth import get_password_hash


# --- User CRUD ---

def create_user(
    db: Session,
    email: str,
    password: str,
    full_name: Optional[str] = None,
    role: str = "user"
) -> User:
    """Create a new user."""
    user = User(
        email=email,
        password_hash=get_password_hash(password),
        full_name=full_name,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> Tuple[List[User], int]:
    """Get all users with pagination."""
    total = db.query(User).count()
    users = db.query(User).offset(skip).limit(limit).all()
    return users, total


def update_user(
    db: Session,
    user_id: int,
    full_name: Optional[str] = None,
    is_active: Optional[bool] = None,
    role: Optional[str] = None
) -> Optional[User]:
    """Update a user's information."""
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    
    if full_name is not None:
        user.full_name = full_name
    if is_active is not None:
        user.is_active = is_active
    if role is not None:
        user.role = role
    
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user_id: int) -> bool:
    """Delete a user."""
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True


# --- Prediction CRUD ---

def create_prediction(
    db: Session,
    user_id: int,
    image_filename: str,
    result: str,
    confidence: float,
    message: Optional[str] = None
) -> Prediction:
    """Create a new prediction record."""
    prediction = Prediction(
        user_id=user_id,
        image_filename=image_filename,
        result=result,
        confidence=confidence,
        message=message
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_user_predictions(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> Tuple[List[Prediction], int]:
    """Get all predictions for a specific user."""
    query = db.query(Prediction).filter(Prediction.user_id == user_id)
    total = query.count()
    predictions = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    return predictions, total


def get_all_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[int] = None,
    result_filter: Optional[str] = None
) -> Tuple[List[Prediction], int]:
    """Get all predictions with optional filters (admin only)."""
    query = db.query(Prediction)
    
    if user_id:
        query = query.filter(Prediction.user_id == user_id)
    if result_filter:
        query = query.filter(Prediction.result == result_filter)
    
    total = query.count()
    predictions = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    return predictions, total


def get_stats(db: Session) -> dict:
    """Get statistics for admin dashboard."""
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_predictions = db.query(Prediction).count()
    pneumonia_count = db.query(Prediction).filter(Prediction.result == "Pneumonia").count()
    normal_count = db.query(Prediction).filter(Prediction.result == "Normal").count()
    predictions_today = db.query(Prediction).filter(Prediction.created_at >= today_start).count()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_predictions": total_predictions,
        "pneumonia_count": pneumonia_count,
        "normal_count": normal_count,
        "predictions_today": predictions_today
    }
