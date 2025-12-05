"""
Streamlit Deployment App for YOLO Farm Detection Model
Detects: Soil, Healthy Crops, and Unhealthy Crops from Drone Imagery
Developed by UENR-ATPS-IDRC
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="UENR Farm Detection System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .header-org {
        color: #ffd700;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 2px;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 3.5em;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: white;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        border-radius: 8px;
        padding: 0 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Upload section */
    .uploadedFile {
        border: 3px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        background: white;
    }
    
    /* Image containers */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #667eea;
    }
    
    /* Legend styling */
    .legend-item {
        display: flex;
        align-items: center;
        padding: 10px;
        margin: 8px 0;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    
    .legend-color {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin-right: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-top: 3rem;
        color: white;
    }
    
    /* Download button special styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# Class information
CLASS_NAMES = {
    0: 'Soil',
    1: 'Healthy Crop',
    2: 'Unhealthy Crop',
    3: 'Other'
}

CLASS_COLORS = {
    0: (139, 69, 19),      # Brown for soil
    1: (34, 139, 34),      # Green for healthy
    2: (255, 69, 0),       # Red for unhealthy
    3: (128, 128, 128)     # Gray for other
}

CLASS_EMOJIS = {
    0: '🟤',
    1: '🟢',
    2: '🔴',
    3: '⚪'
}


@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_image(image, model):
    """Process single image and return results"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run inference with fixed parameters
    results = model.predict(
        img_array,
        conf=0.25,
        iou=0.45,
        verbose=False
    )
    
    return results[0]


def draw_detections(image, results):
    """Draw bounding boxes on image"""
    img_array = np.array(image)
    
    # Get boxes, classes, and confidences
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    # Draw each detection
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        # Draw rectangle with thicker lines
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 4)
        
        # Prepare label
        label = f"{CLASS_NAMES.get(cls, f'Class {cls}')} {conf:.2f}"
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_array, (x1, y1 - 35), (x1 + w + 10, y1), color, -1)
        cv2.putText(img_array, label, (x1 + 5, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return Image.fromarray(img_array)


def get_detection_stats(results):
    """Calculate detection statistics"""
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    # Count detections per class
    class_counts = {}
    for cls in classes:
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Calculate average confidence per class
    class_conf = {}
    for cls in set(classes):
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        cls_confidences = confidences[classes == cls]
        class_conf[class_name] = np.mean(cls_confidences)
    
    return class_counts, class_conf


def create_stats_charts(class_counts, class_conf):
    """Create visualization charts for detection statistics"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection counts bar chart
        if class_counts:
            df_counts = pd.DataFrame(list(class_counts.items()), 
                                    columns=['Class', 'Count'])
            fig_counts = px.bar(df_counts, x='Class', y='Count',
                              title='🔢 Detection Counts by Class',
                              color='Class',
                              color_discrete_map={
                                  'Soil': '#8B4513',
                                  'Healthy Crop': '#228B22',
                                  'Unhealthy Crop': '#FF4500',
                                  'Other': '#808080'
                              })
            fig_counts.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, color='#2c3e50')
            )
            st.plotly_chart(fig_counts, use_container_width=True)
        else:
            st.info("No detections found")
    
    with col2:
        # Confidence scores bar chart
        if class_conf:
            df_conf = pd.DataFrame(list(class_conf.items()), 
                                  columns=['Class', 'Avg Confidence'])
            fig_conf = px.bar(df_conf, x='Class', y='Avg Confidence',
                            title='📊 Average Confidence by Class',
                            color='Class',
                            color_discrete_map={
                                'Soil': '#8B4513',
                                'Healthy Crop': '#228B22',
                                'Unhealthy Crop': '#FF4500',
                                'Other': '#808080'
                            })
            fig_conf.update_layout(
                showlegend=False, 
                yaxis_range=[0, 1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, color='#2c3e50')
            )
            st.plotly_chart(fig_conf, use_container_width=True)


def process_video(video_path, model, progress_bar):
    """Process video file and return annotated video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_class_counts = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            frame,
            conf=0.25,
            iou=0.45,
            verbose=False
        )[0]
        
        # Draw detections
        annotated_frame = results.plot()
        
        # Count detections
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            class_name = CLASS_NAMES.get(cls, f"Class {cls}")
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    return output_path, all_class_counts


def main():
    # Beautiful Header
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🌾 Farm Detection System</div>
            <div class="header-subtitle">AI-Powered Crop Health Analysis from Drone Imagery</div>
            <div class="header-org">UENR • ATPS • IDRC</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Class Legend in sidebar (minimized)
    with st.sidebar:
        st.markdown("### 📋 Detection Classes")
        for cls_id, cls_name in CLASS_NAMES.items():
            emoji = CLASS_EMOJIS[cls_id]
            st.markdown(f"{emoji} **{cls_name}**")
        
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <p style='font-size: 0.9rem; color: #666;'>
                    <b>Developed by</b><br>
                    University of Energy and<br>
                    Natural Resources (UENR)<br>
                    African Technology Policy<br>
                    Studies Network (ATPS)<br>
                    International Development<br>
                    Research Centre (IDRC)
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please ensure the model file 'best.pt' is in the same directory.")
        return
    
    with st.spinner("🔄 Initializing AI Model..."):
        model = load_model(model_path)
    
    if model is None:
        return
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    # Tab 1: Image Detection
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_upload, col_info = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "📤 Upload Drone Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a drone image of your farm for analysis"
            )
        
        with col_info:
            st.info("""
                **📝 Instructions:**
                1. Upload a drone image
                2. Wait for AI analysis
                3. View detection results
                4. Download annotated image
            """)
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📸 Original Image")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Process image
            with st.spinner("🔍 Analyzing image with AI..."):
                results = process_image(image, model)
                annotated_image = draw_detections(image, results)
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(annotated_image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistics
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📊 Analysis Results")
            
            class_counts, class_conf = get_detection_stats(results)
            
            # Display metrics in beautiful cards
            if class_counts:
                cols = st.columns(len(class_counts))
                for idx, (class_name, count) in enumerate(class_counts.items()):
                    with cols[idx]:
                        # Find emoji for this class
                        emoji = '🔹'
                        for cls_id, name in CLASS_NAMES.items():
                            if name == class_name:
                                emoji = CLASS_EMOJIS[cls_id]
                                break
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <h2 style='text-align: center; margin: 0;'>{emoji}</h2>
                                <h3 style='text-align: center; margin: 10px 0;'>{class_name}</h3>
                                <h1 style='text-align: center; color: #667eea; margin: 10px 0;'>{count}</h1>
                                <p style='text-align: center; color: #666; margin: 0;'>
                                    {class_conf.get(class_name, 0):.1%} confidence
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No objects detected in this image.")
            
            # Charts
            if class_counts:
                st.markdown("<br>", unsafe_allow_html=True)
                create_stats_charts(class_counts, class_conf)
            
            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Save annotated image
            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            annotated_image.save(img_bytes.name)
            
            col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
            with col_download2:
                with open(img_bytes.name, 'rb') as f:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=f,
                        file_name="farm_detection_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    # Tab 2: Video Detection
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_upload, col_info = st.columns([2, 1])
        
        with col_upload:
            uploaded_video = st.file_uploader(
                "📤 Upload Drone Video",
                type=['mp4', 'avi', 'mov'],
                help="Upload a drone video of your farm for analysis"
            )
        
        with col_info:
            st.info("""
                **📝 Instructions:**
                1. Upload a drone video
                2. Click 'Process Video'
                3. Wait for AI analysis
                4. Download processed video
            """)
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display original video
            st.markdown("### 📹 Original Video")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.video(tfile.name)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Process video
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                process_btn = st.button("🎬 Process Video", use_container_width=True)
            
            if process_btn:
                st.markdown("### 🔄 Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Analyzing video frames with AI..."):
                    output_path, class_counts = process_video(
                        tfile.name, model, progress_bar
                    )
                
                status_text.empty()
                st.success("✅ Video processing completed successfully!")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display processed video
                st.markdown("### 🎯 Processed Video")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.video(output_path)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Statistics
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## 📊 Video Analysis Results")
                
                if class_counts:
                    cols = st.columns(len(class_counts))
                    for idx, (class_name, count) in enumerate(class_counts.items()):
                        with cols[idx]:
                            # Find emoji for this class
                            emoji = '🔹'
                            for cls_id, name in CLASS_NAMES.items():
                                if name == class_name:
                                    emoji = CLASS_EMOJIS[cls_id]
                                    break
                            
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style='text-align: center; margin: 0;'>{emoji}</h2>
                                    <h3 style='text-align: center; margin: 10px 0;'>{class_name}</h3>
                                    <h1 style='text-align: center; color: #667eea; margin: 10px 0;'>{count}</h1>
                                    <p style='text-align: center; color: #666; margin: 0;'>
                                        Total Detections
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Download processed video
                st.markdown("<br>", unsafe_allow_html=True)
                col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
                with col_download2:
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name="farm_detection_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="footer">
            <h3 style='margin-bottom: 1rem;'>🌾 Farm Detection System</h3>
            <p style='font-size: 1.1rem; margin-bottom: 1rem;'>
                Empowering Precision Agriculture with Artificial Intelligence
            </p>
            <p style='font-size: 0.95rem; opacity: 0.9;'>
                <b>Developed by:</b><br>
                University of Energy and Natural Resources (UENR) •
                African Technology Policy Studies Network (ATPS) •
                International Development Research Centre (IDRC)
            </p>
            <p style='margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;'>
                © 2024 UENR-ATPS-IDRC | All Rights Reserved
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
