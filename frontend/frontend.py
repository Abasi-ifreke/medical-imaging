import streamlit as st
import requests
from PIL import Image
import io
import os

# Backend URL - can be overridden by environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Check backend connectivity
try:
    response = requests.get(API_URL.replace("/predict", "/docs"))  # Test connection
    print("✅ Connection Successful:", response.status_code)
except requests.exceptions.RequestException as e:
    print("❌ Connection Failed:", e)

# Streamlit UI
st.title("🩺 Medical Diagnosis - Pneumonia Detection")
st.write("Upload an X-ray image to verify and diagnose:")

uploaded_file = st.file_uploader("📷 Upload an X-ray image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Convert to byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(API_URL, files={"file": ("image.png", img_byte_arr, "image/png")})

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("🧠 Diagnostic Results:")
                    
                    # Display the message if available
                    if result.get('message'):
                        if result.get('prediction') == "Pneumonia":
                            st.error(f"⚠️ {result['message']}")
                        elif result.get('prediction') == "Normal":
                            st.success(f"✅ {result['message']}")
                        else:
                            st.warning(f"ℹ️ {result['message']}")
                    
                    # Display diagnosis if available
                    if result.get('prediction'):
                        st.write(f"**Diagnosis:** **{result['prediction']}**")
                        
                        # Display confidence score with progress bar
                        if result.get('confidence') is not None:
                            confidence_percent = result['confidence'] * 100
                            st.write(f"**Confidence:** {confidence_percent:.1f}%")
                            st.progress(result['confidence'])

                else:
                    st.error("❌ Error: Backend failed to process the image.")
            except Exception as e:
                st.error(f"🔌 Connection error: {e}")
