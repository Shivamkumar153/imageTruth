import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime

# Set page config with dark theme
st.set_page_config(
    page_title="ImageTruth",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
.uploadedFile {
    background-color: #262730;
    border-radius: 10px;
    padding: 20px;
}
.css-1d391kg {
    background-color: #262730;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("ImageTruth")
st.write("Upload an image to verify its authenticity")

# Image Requirements in a container with custom styling
st.markdown("""
<div style='background-color: #1a2634; padding: 20px; border-radius: 10px; margin: 10px 0;'>
    <h2 style='color: #ffffff;'>Image Requirements:</h2>
    <ul style='color: #4a9eff;'>
        <li>Face should be clearly visible</li>
        <li>Good lighting conditions</li>
        <li>Minimum size: 96x96 pixels</li>
        <li>Supported formats: JPG, JPEG, PNG</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Load the trained model
try:
    model = load_model('deepfake_detection_model.h5')
except:
    st.error("Error: Could not load model file. Please make sure 'deepfake_detection_model.h5' exists in the same directory.")
    st.stop()

# File uploader with custom styling
st.markdown("<p style='color: #718096;'>Choose an image...</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

# Display drag and drop area with custom styling
if not uploaded_file:
    st.markdown("""
    <div style='background-color: #262730; padding: 30px; border-radius: 10px; text-align: center; margin: 10px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/685/685686.png' style='width: 50px; opacity: 0.5;'>
        <p style='color: #718096; margin-top: 10px;'>Drag and drop file here</p>
        <p style='color: #4a9eff; font-size: 0.8em;'>Limit 200MB per file • JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((96, 96))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    result = prediction[0]
    
    # Display result
    st.write("## Analysis Result")
    is_fake = np.argmax(result) == 0
    confidence = result[0]*100 if is_fake else result[1]*100
    
    if is_fake:
        st.error("⚠️ This image appears to be FAKE")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.success("✅ This image appears to be REAL")
        st.write(f"Confidence: {confidence:.2f}%")
    
    # Generate report
    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": uploaded_file.name,
        "result": "FAKE" if is_fake else "REAL",
        "confidence": f"{confidence:.2f}%"
    }
    
    # Create download button for report
    report_json = json.dumps(report, indent=2)
    st.download_button(
        label="📥 Download Analysis Report",
        data=report_json,
        file_name="image_analysis_report.json",
        mime="application/json"
    )

# Add instructions
st.markdown("""
### Image Requirements:
- Face should be clearly visible
- Good lighting conditions
- Minimum size: 96x96 pixels
- Supported formats: JPG, JPEG, PNG""")

# Footer section
st.markdown("""
---
**Contact Us:**
For more information and queries, please contact us at [contact@example.com](mailto:contact@example.com).

**Follow us on:**
[Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com) | [Facebook](https://facebook.com)
""")
