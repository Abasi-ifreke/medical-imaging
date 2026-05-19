import streamlit as st
import requests
import os
import pandas as pd

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
LOGIN_URL = f"{API_BASE_URL}/auth/login"
ME_URL = f"{API_BASE_URL}/auth/me"
ADMIN_USERS_URL = f"{API_BASE_URL}/admin/users"
ADMIN_PREDICTIONS_URL = f"{API_BASE_URL}/admin/predictions"
ADMIN_STATS_URL = f"{API_BASE_URL}/admin/stats"

st.set_page_config(
    page_title="Admin Panel - Medical Imaging",
    page_icon="🔧",
    layout="wide"
)

if "admin_token" not in st.session_state:
    st.session_state.admin_token = None
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None


def get_auth_headers():
    """Get authorization headers if logged in."""
    if st.session_state.admin_token:
        return {"Authorization": f"Bearer {st.session_state.admin_token}"}
    return {}


def admin_login(email: str, password: str) -> bool:
    """Attempt to login with admin credentials."""
    try:
        response = requests.post(
            LOGIN_URL,
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.admin_token = data["access_token"]
            
            me_response = requests.get(
                ME_URL,
                headers={"Authorization": f"Bearer {data['access_token']}"}
            )
            if me_response.status_code == 200:
                user = me_response.json()
                if user.get("role") != "admin":
                    st.error("Access denied. Admin privileges required.")
                    st.session_state.admin_token = None
                    return False
                st.session_state.admin_user = user
                return True
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def admin_logout():
    """Clear admin session."""
    st.session_state.admin_token = None
    st.session_state.admin_user = None


def fetch_stats():
    """Fetch platform statistics."""
    try:
        response = requests.get(ADMIN_STATS_URL, headers=get_auth_headers())
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def fetch_users(skip=0, limit=50):
    """Fetch all users."""
    try:
        response = requests.get(
            ADMIN_USERS_URL,
            headers=get_auth_headers(),
            params={"skip": skip, "limit": limit}
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"users": [], "total": 0}


def fetch_predictions(skip=0, limit=50, user_id=None, result_filter=None):
    """Fetch all predictions."""
    try:
        params = {"skip": skip, "limit": limit}
        if user_id:
            params["user_id"] = user_id
        if result_filter:
            params["result_filter"] = result_filter
        
        response = requests.get(
            ADMIN_PREDICTIONS_URL,
            headers=get_auth_headers(),
            params=params
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"predictions": [], "total": 0}


def update_user(user_id: int, updates: dict):
    """Update a user."""
    try:
        response = requests.put(
            f"{ADMIN_USERS_URL}/{user_id}",
            headers=get_auth_headers(),
            json=updates
        )
        return response.status_code == 200
    except Exception:
        return False


def delete_user(user_id: int):
    """Delete a user."""
    try:
        response = requests.delete(
            f"{ADMIN_USERS_URL}/{user_id}",
            headers=get_auth_headers()
        )
        return response.status_code == 200
    except Exception:
        return False


def show_login_page():
    """Display the admin login page."""
    st.title("🔧 Admin Panel - Medical Imaging")
    st.write("Please log in with your admin credentials.")
    
    with st.form("admin_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted and email and password:
            if admin_login(email, password):
                st.rerun()


def show_dashboard():
    """Display the admin dashboard."""
    st.title("🔧 Admin Panel")
    
    st.sidebar.title("👤 Admin")
    st.sidebar.write(f"**{st.session_state.admin_user.get('email')}**")
    if st.sidebar.button("Logout"):
        admin_logout()
        st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "👥 Users", "📋 Predictions"])
    
    with tab1:
        show_stats_tab()
    
    with tab2:
        show_users_tab()
    
    with tab3:
        show_predictions_tab()


def show_stats_tab():
    """Display statistics dashboard."""
    st.header("Platform Statistics")
    
    stats = fetch_stats()
    if not stats:
        st.error("Failed to load statistics.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", stats["total_users"])
        st.metric("Active Users", stats["active_users"])
    
    with col2:
        st.metric("Total Predictions", stats["total_predictions"])
        st.metric("Predictions Today", stats["predictions_today"])
    
    with col3:
        st.metric("Pneumonia Cases", stats["pneumonia_count"])
        st.metric("Normal Cases", stats["normal_count"])
    
    st.divider()
    
    st.subheader("Prediction Distribution")
    if stats["total_predictions"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            chart_data = pd.DataFrame({
                "Result": ["Pneumonia", "Normal"],
                "Count": [stats["pneumonia_count"], stats["normal_count"]]
            })
            st.bar_chart(chart_data.set_index("Result"))
        
        with col2:
            pneumonia_pct = (stats["pneumonia_count"] / stats["total_predictions"]) * 100
            normal_pct = (stats["normal_count"] / stats["total_predictions"]) * 100
            st.write(f"**Pneumonia Rate:** {pneumonia_pct:.1f}%")
            st.write(f"**Normal Rate:** {normal_pct:.1f}%")
    else:
        st.info("No predictions yet.")


def show_users_tab():
    """Display users management."""
    st.header("User Management")
    
    data = fetch_users()
    users = data.get("users", [])
    total = data.get("total", 0)
    
    st.write(f"Total users: **{total}**")
    
    if not users:
        st.info("No users found.")
        return
    
    for user in users:
        with st.expander(f"{'🔑' if user['role'] == 'admin' else '👤'} {user['email']} - {user['role'].capitalize()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ID:** {user['id']}")
                st.write(f"**Name:** {user.get('full_name') or 'N/A'}")
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Role:** {user['role']}")
                st.write(f"**Active:** {'Yes' if user['is_active'] else 'No'}")
                st.write(f"**Created:** {user['created_at'][:10]}")
            
            with col2:
                if user['id'] != st.session_state.admin_user['id']:
                    st.write("**Actions:**")
                    
                    new_active = not user['is_active']
                    if st.button(
                        f"{'Activate' if new_active else 'Deactivate'}",
                        key=f"toggle_{user['id']}"
                    ):
                        if update_user(user['id'], {"is_active": new_active}):
                            st.success("User updated!")
                            st.rerun()
                        else:
                            st.error("Failed to update user.")
                    
                    new_role = "admin" if user['role'] == "user" else "user"
                    if st.button(
                        f"Make {new_role.capitalize()}",
                        key=f"role_{user['id']}"
                    ):
                        if update_user(user['id'], {"role": new_role}):
                            st.success("User role updated!")
                            st.rerun()
                        else:
                            st.error("Failed to update user.")
                    
                    if st.button("Delete", key=f"delete_{user['id']}", type="secondary"):
                        if delete_user(user['id']):
                            st.success("User deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete user.")
                else:
                    st.info("Cannot modify your own account.")


def show_predictions_tab():
    """Display predictions list."""
    st.header("Prediction History")
    
    col1, col2 = st.columns(2)
    with col1:
        result_filter = st.selectbox(
            "Filter by result",
            ["All", "Pneumonia", "Normal"]
        )
    with col2:
        user_filter = st.text_input("Filter by user ID")
    
    filter_result = result_filter if result_filter != "All" else None
    filter_user = int(user_filter) if user_filter.isdigit() else None
    
    data = fetch_predictions(
        limit=100,
        user_id=filter_user,
        result_filter=filter_result
    )
    predictions = data.get("predictions", [])
    total = data.get("total", 0)
    
    st.write(f"Showing {len(predictions)} of {total} predictions")
    
    if not predictions:
        st.info("No predictions found.")
        return
    
    df = pd.DataFrame([
        {
            "ID": p["id"],
            "User": p.get("user_email", f"User {p['user_id']}"),
            "File": p["image_filename"],
            "Result": p["result"],
            "Confidence": f"{p['confidence']*100:.1f}%",
            "Date": p["created_at"][:10]
        }
        for p in predictions
    ])
    
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    st.subheader("Detailed View")
    for pred in predictions[:10]:
        result_icon = "🔴" if pred["result"] == "Pneumonia" else "🟢"
        with st.expander(f"{result_icon} {pred['result']} - {pred.get('user_email', 'Unknown')} - {pred['created_at'][:10]}"):
            st.write(f"**Prediction ID:** {pred['id']}")
            user_display = pred.get(
                "user_email",
                f"User ID: {pred['user_id']}"
            )

            st.write(f"**User:** {user_display}") 
            st.write(f"**File:** {pred['image_filename']}")
            st.write(f"**Result:** {pred['result']}")
            st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
            st.write(f"**Message:** {pred.get('message', 'N/A')}")
            st.write(f"**Date:** {pred['created_at']}")


if st.session_state.admin_token and st.session_state.admin_user:
    show_dashboard()
else:
    show_login_page()
