import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Model filename and Google Drive file ID
MODEL_FILE = "final_epoch13_855acc_fullmodel.h5"
DRIVE_ID = "1gwCm1YTUHxLtGl6Gj5Ck4RXvRj3M37am"

# Download model if not already present
if not os.path.exists(MODEL_FILE):
    st.info("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load model
@st.cache_resource
def load_dr_model():
    return load_model(MODEL_FILE)

model = load_dr_model()
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

# Title
st.title("🩺 Diabetic Retinopathy Detector")
st.markdown("Upload a retina image to detect DR stage using AI")

# Upload
uploaded_file = st.file_uploader("Choose a retina image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Retina Preview', use_column_width=True)

    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
            st.success(f"🧠 Prediction: **{prediction}**")
            st.bar_chart(preds[0])
