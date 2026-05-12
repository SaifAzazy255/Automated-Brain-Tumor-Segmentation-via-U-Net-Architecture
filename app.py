import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- 1) Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="centered"
)

# --- 2) Model Loading ---
@st.cache_resource
def load_segmentation_model():
    model_path = 'model.keras'
    if not os.path.exists(model_path):
        return None
    try:
        # Using compile=False is safer for inference across different environments
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_segmentation_model()

# --- 3) User Interface ---
st.title("🧠 Brain Tumor Segmentation App")
st.markdown("""
### Smart Systems Engineering Project
This application utilizes a **U-Net** architecture to detect and segment brain tumor regions from MRI scans.
""")

# Support for multiple medical formats
uploaded_file = st.file_uploader("Upload MRI Scan (JPG, PNG, TIF)...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    try:
        # Processing image to be displayed correctly
        raw_image = Image.open(uploaded_file)
        
        # Standardize to RGB for display and processing
        if raw_image.mode != 'RGB':
            image = raw_image.convert('RGB')
        else:
            image = raw_image
            
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
        if st.button('Run AI Segmentation'):
            if model is not None:
                with st.spinner('Analyzing image and extracting tumor features...'):
                    # --- 4) Preprocessing Stage ---
                    img_array = np.array(image)
                    img_resized = cv2.resize(img_array, (256, 256))
                    img_normalized = img_resized / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)
                    
                    # --- 5) Prediction Stage ---
                    prediction = model.predict(img_input)
                    # Squeeze to get (256, 256, 1) or (256, 256)
                    mask = (prediction[0] > 0.5).astype(np.uint8)
                    mask = np.squeeze(mask)
                    
                    # --- 6) Visualization Stage ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Predicted Mask")
                        # Show Binary Mask
                        st.image(mask * 255, use_container_width=True)
                    
                    with col2:
                        st.subheader("Overlay View")
                        # Red mask for tumor
                        colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
                        colored_mask[mask == 1] = [255, 0, 0] 
                        
                        # Blend 70% original, 30% mask
                        overlay = cv2.addWeighted(img_resized, 0.7, colored_mask, 0.3, 0)
                        st.image(overlay, use_container_width=True)
                    
                    st.success("Analysis Complete!")
            else:
                st.error("Model file 'model.keras' not found. Please place it in the project folder.")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Sidebar Information
st.sidebar.info(f"""
**Technical Details:**
- Model Architecture: U-Net
- Target Metric: Dice 0.86 / IoU 0.75
- Input Resolution: 256x256
""")