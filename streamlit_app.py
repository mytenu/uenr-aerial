"""
Smart Farm Monitor - AI-Powered Precision Agriculture
Developed by UENR, ATPS & IDRC (2025)
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import torch
import plotly.express as px
import pandas as pd
import base64

# Page configuration
st.set_page_config(
    page_title="Smart Farm Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to convert image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Class information
CLASS_NAMES = {0: 'Soil', 1: 'Healthy Crop', 2: 'Unhealthy Crop', 3: 'Other'}
CLASS_COLORS = {0: (139, 69, 19), 1: (34, 139, 34), 2: (255, 69, 0), 3: (128, 128, 128)}
CLASS_COLORS_HEX = {0: '#8B4513', 1: '#228B22', 2: '#FF4500', 3: '#808080'}

# Load YOLOv5 model via torch hub
@st.cache_resource
def load_model(model_path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.conf = 0.25
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Run detection
def process_image(image, model):
    img_array = np.array(image)
    results = model(img_array)
    return results

# Draw boxes (YOLOv5 format)
def draw_detections(image, results):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

        label = f"{CLASS_NAMES.get(cls, f'Class {cls}')} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h - 10, x1 + text_w + 10, y1], fill=color)
        draw.text((x1 + 5, y1 - text_h - 5), label, fill=(255, 255, 255), font=font)

    return img

# Stats
def get_detection_stats(results):
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) == 0:
        return {}, {}

    classes = detections[:, 5].astype(int)
    confidences = detections[:, 4]

    class_counts = {}
    class_conf = {}

    for cls in classes:
        cname = CLASS_NAMES.get(cls, f"Class {cls}")
        class_counts[cname] = class_counts.get(cname, 0) + 1

    for cls in set(classes):
        cname = CLASS_NAMES.get(cls, f"Class {cls}")
        class_conf[cname] = float(confidences[classes == cls].mean())

    return class_counts, class_conf

# Main app
def main():
    bg_image = get_base64_image("aerial1.jpeg")

    st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #2E7D32; text-align: center; font-weight: bold;}
    .sub-header { font-size: 1.2rem; text-align: center; margin-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🌾 Smart Farm Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nutrient Deficiency Detection in Cashew Farms Using Aerial Imagery</p>', unsafe_allow_html=True)

    model_path = "best.pt"

    if not os.path.exists(model_path):
        st.error("🚨 Model not found. Place best.pt in app directory.")
        return

    model = load_model(model_path)
    if model is None:
        return

    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "ℹ️ About"])

    # Image Tab
    with tab1:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    results = process_image(image, model)
                    annotated = draw_detections(image, results)
                    st.image(annotated, use_container_width=True)

                    num_det = len(results.xyxy[0])
                    st.caption(f"✅ Detected {num_det} objects")

            st.subheader("📊 Detection Statistics")
            class_counts, class_conf = get_detection_stats(results)

            if class_counts:
                cols = st.columns(len(class_counts))
                for i, (name, count) in enumerate(class_counts.items()):
                    with cols[i]:
                        st.metric(name, count, f"{class_conf.get(name,0)*100:.1f}% conf")

                df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                st.plotly_chart(px.bar(df, x="Class", y="Count"), use_container_width=True)

            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            annotated.save(img_bytes.name)
            with open(img_bytes.name, "rb") as f:
                st.download_button("📥 Download Annotated Image", f, "result.png", "image/png")

    # Video tab
    with tab2:
        st.warning("Video detection currently unavailable.")

    # About tab
    with tab3:
        st.write("""
        Smart Farm Monitor helps cashew farmers detect nutrient deficiencies
        using AI-based aerial image analysis.
        """)

if __name__ == "__main__":
    main()
