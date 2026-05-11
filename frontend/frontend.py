import streamlit as st
import requests
from PIL import Image
import io
import os

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
LOGIN_URL = f"{API_BASE_URL}/auth/login"
REGISTER_URL = f"{API_BASE_URL}/auth/register"
ME_URL = f"{API_BASE_URL}/auth/me"
PREDICTIONS_URL = f"{API_BASE_URL}/predictions"

st.set_page_config(
    page_title="Medical Imaging - Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None


def get_auth_headers():
    """Get authorization headers if logged in."""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def login(email: str, password: str) -> bool:
    """Attempt to login with email and password."""
    try:
        response = requests.post(
            LOGIN_URL,
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            fetch_user_profile()
            return True
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def register(email: str, password: str, full_name: str) -> bool:
    """Register a new user account."""
    try:
        response = requests.post(
            REGISTER_URL,
            json={"email": email, "password": password, "full_name": full_name}
        )
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
            return True
        else:
            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def fetch_user_profile():
    """Fetch the current user's profile."""
    try:
        response = requests.get(ME_URL, headers=get_auth_headers())
        if response.status_code == 200:
            st.session_state.user = response.json()
        else:
            logout()
    except Exception:
        logout()


def logout():
    """Clear session and logout."""
    st.session_state.token = None
    st.session_state.user = None


def fetch_prediction_history():
    """Fetch the user's prediction history."""
    try:
        response = requests.get(
            PREDICTIONS_URL,
            headers=get_auth_headers(),
            params={"limit": 10}
        )
        if response.status_code == 200:
            return response.json().get("predictions", [])
    except Exception:
        pass
    return []


def show_login_page():
    """Display the login/register page."""
    st.title("🩺 Medical Imaging - Pneumonia Detection")
    st.write("Please log in or register to use the diagnostic tool.")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            
            if submitted and email and password:
                if login(email, password):
                    st.rerun()
    
    with tab2:
        with st.form("register_form"):
            full_name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if not email or not password:
                    st.error("Email and password are required.")
                elif password != password_confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    if register(email, password, full_name):
                        st.rerun()


def show_main_app():
    """Display the main application."""
    user = st.session_state.user
    
    st.sidebar.title("👤 User Profile")
    st.sidebar.write(f"**{user.get('full_name') or user.get('email')}**")
    st.sidebar.write(f"Email: {user.get('email')}")
    st.sidebar.write(f"Role: {user.get('role', 'user').capitalize()}")
    
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📊 Recent Predictions")
    history = fetch_prediction_history()
    if history:
        for pred in history[:5]:
            result_icon = "🔴" if pred["result"] == "Pneumonia" else "🟢"
            st.sidebar.write(f"{result_icon} {pred['result']} ({pred['confidence']*100:.1f}%)")
            st.sidebar.caption(pred["created_at"][:10])
    else:
        st.sidebar.write("No predictions yet.")
    
    st.title("🩺 Medical Diagnosis - Pneumonia Detection")
    st.write("Upload an X-ray image to verify and diagnose:")
    
    uploaded_file = st.file_uploader("📷 Upload an X-ray image (JPG/PNG)...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Diagnose", use_container_width=True):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                
                with st.spinner("Analyzing image..."):
                    try:
                        response = requests.post(
                            PREDICT_URL,
                            files={"file": (uploaded_file.name, img_byte_arr, "image/png")},
                            headers=get_auth_headers()
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.subheader("🧠 Diagnostic Results:")
                            
                            if result.get('message'):
                                if result.get('prediction') == "Pneumonia":
                                    st.error(f"⚠️ {result['message']}")
                                elif result.get('prediction') == "Normal":
                                    st.success(f"✅ {result['message']}")
                                else:
                                    st.warning(f"ℹ️ {result['message']}")
                            
                            if result.get('prediction'):
                                st.write(f"**Diagnosis:** **{result['prediction']}**")
                                
                                if result.get('confidence') is not None:
                                    confidence_percent = result['confidence'] * 100
                                    st.write(f"**Confidence:** {confidence_percent:.1f}%")
                                    st.progress(result['confidence'])
                            
                            if result.get('prediction_id'):
                                st.caption(f"Prediction saved (ID: {result['prediction_id']})")
                        
                        elif response.status_code == 401:
                            st.error("Session expired. Please log in again.")
                            logout()
                            st.rerun()
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Backend failed to process the image.')}")
                    
                    except Exception as e:
                        st.error(f"🔌 Connection error: {e}")
    
    st.divider()
    
    st.subheader("📋 Prediction History")
    if history:
        for pred in history:
            with st.expander(f"{pred['result']} - {pred['created_at'][:10]} ({pred['confidence']*100:.1f}%)"):
                st.write(f"**File:** {pred['image_filename']}")
                st.write(f"**Result:** {pred['result']}")
                st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
                st.write(f"**Message:** {pred.get('message', 'N/A')}")
                st.write(f"**Date:** {pred['created_at']}")
    else:
        st.info("No prediction history yet. Upload an image to get started!")


if st.session_state.token and not st.session_state.user:
    fetch_user_profile()

if st.session_state.token and st.session_state.user:
    show_main_app()
else:
    show_login_page()
