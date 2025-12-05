"""
Streamlit Deployment App for YOLO Farm Detection Model
Detects: Soil, Healthy Crops, and Unhealthy Crops from Drone Imagery
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
    page_title="Farm Detection System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    h3 {
        color: #4CAF50;
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


@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_image(image, model, conf_threshold, iou_threshold):
    """Process single image and return results"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model.predict(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    return results[0]


def draw_detections(image, results, show_labels=True, show_conf=True):
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
        
        # Draw rectangle
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        label = ""
        if show_labels:
            label = CLASS_NAMES.get(cls, f"Class {cls}")
        if show_conf:
            label += f" {conf:.2f}" if label else f"{conf:.2f}"
        
        # Draw label background
        if label:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_array, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(img_array, label, (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
                              title='Detection Counts by Class',
                              color='Class',
                              color_discrete_map={
                                  'Soil': 'brown',
                                  'Healthy Crop': 'green',
                                  'Unhealthy Crop': 'red',
                                  'Other': 'gray'
                              })
            fig_counts.update_layout(showlegend=False)
            st.plotly_chart(fig_counts, use_container_width=True)
        else:
            st.info("No detections found")
    
    with col2:
        # Confidence scores bar chart
        if class_conf:
            df_conf = pd.DataFrame(list(class_conf.items()), 
                                  columns=['Class', 'Avg Confidence'])
            fig_conf = px.bar(df_conf, x='Class', y='Avg Confidence',
                            title='Average Confidence by Class',
                            color='Class',
                            color_discrete_map={
                                'Soil': 'brown',
                                'Healthy Crop': 'green',
                                'Unhealthy Crop': 'red',
                                'Other': 'gray'
                            })
            fig_conf.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_conf, use_container_width=True)


def process_video(video_path, model, conf_threshold, iou_threshold, progress_bar):
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
            conf=conf_threshold,
            iou=iou_threshold,
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
    # Header
    st.markdown("<h1 style='text-align: center;'>🌾 Farm Detection System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>AI-Powered Crop Health Analysis from Drone Imagery</p>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/drone.png", width=100)
        st.markdown("## ⚙️ Configuration")
        
        # Model selection
        st.markdown("### Model Settings")
        model_path = st.text_input(
            "Model Path",
            value="farm_detection/yolov8n_farm/weights/best.pt",
            help="Path to your trained YOLO model"
        )
        
        # Detection parameters
        st.markdown("### Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        # Display options
        st.markdown("### Display Options")
        show_labels = st.checkbox("Show Labels", value=True)
        show_conf = st.checkbox("Show Confidence", value=True)
        
        # Class legend
        st.markdown("### 📋 Class Legend")
        for cls_id, cls_name in CLASS_NAMES.items():
            color = CLASS_COLORS[cls_id]
            st.markdown(
                f"<div style='background-color: rgb{color}; padding: 5px; "
                f"border-radius: 5px; margin: 5px 0; color: white;'>"
                f"<b>{cls_name}</b></div>",
                unsafe_allow_html=True
            )
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please check the model path in the sidebar.")
        return
    
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Detection", "🎥 Video Detection", 
                                       "📹 Webcam Detection", "ℹ️ About"])
    
    # Tab 1: Image Detection
    with tab1:
        st.markdown("## Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a drone image of your farm"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner("Detecting..."):
                results = process_image(image, model, conf_threshold, iou_threshold)
                annotated_image = draw_detections(image, results, show_labels, show_conf)
            
            with col2:
                st.markdown("### Detection Results")
                st.image(annotated_image, use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.markdown("## 📊 Detection Statistics")
            
            class_counts, class_conf = get_detection_stats(results)
            
            # Display metrics
            cols = st.columns(len(class_counts) if class_counts else 1)
            for idx, (class_name, count) in enumerate(class_counts.items()):
                with cols[idx]:
                    st.metric(
                        label=class_name,
                        value=f"{count} detected",
                        delta=f"{class_conf.get(class_name, 0):.2%} confidence"
                    )
            
            if not class_counts:
                st.info("No objects detected. Try adjusting the confidence threshold.")
            
            # Charts
            if class_counts:
                st.markdown("---")
                create_stats_charts(class_counts, class_conf)
            
            # Download button
            st.markdown("---")
            # Save annotated image
            img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            annotated_image.save(img_bytes.name)
            
            with open(img_bytes.name, 'rb') as f:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=f,
                    file_name="farm_detection_result.png",
                    mime="image/png"
                )
    
    # Tab 2: Video Detection
    with tab2:
        st.markdown("## Upload Video for Detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a drone video of your farm"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            # Display original video
            st.markdown("### Original Video")
            st.video(tfile.name)
            
            # Process video
            if st.button("🎬 Process Video", key="process_video"):
                st.markdown("### Processing Video...")
                progress_bar = st.progress(0)
                
                with st.spinner("Processing frames..."):
                    output_path, class_counts = process_video(
                        tfile.name, model, conf_threshold, iou_threshold, progress_bar
                    )
                
                st.success("✅ Video processing completed!")
                
                # Display processed video
                st.markdown("### Processed Video")
                st.video(output_path)
                
                # Statistics
                st.markdown("---")
                st.markdown("## 📊 Video Detection Statistics")
                
                cols = st.columns(len(class_counts) if class_counts else 1)
                for idx, (class_name, count) in enumerate(class_counts.items()):
                    with cols[idx]:
                        st.metric(label=class_name, value=f"{count} total")
                
                # Download processed video
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="farm_detection_video.mp4",
                        mime="video/mp4"
                    )
    
    # Tab 3: Webcam Detection
    with tab3:
        st.markdown("## Real-time Webcam Detection")
        st.info("📹 This feature allows real-time detection using your webcam.")
        
        st.markdown("""
        ### Instructions:
        1. Click the button below to enable your webcam
        2. Grant camera permissions when prompted
        3. Point your camera at the target area
        4. Detections will appear in real-time
        """)
        
        enable_webcam = st.checkbox("Enable Webcam", value=False)
        
        if enable_webcam:
            st.warning("⚠️ Webcam feature requires running locally with camera access.")
            st.code("""
# For local deployment with webcam:
# Run this code separately

import cv2
from ultralytics import YOLO

model = YOLO('your_model_path.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()
    
    cv2.imshow('Farm Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            """, language="python")
    
    # Tab 4: About
    with tab4:
        st.markdown("## About This Application")
        
        st.markdown("""
        ### 🌾 Farm Detection System
        
        This application uses YOLOv8 deep learning model to detect and classify:
        - **Soil** - Bare ground areas
        - **Healthy Crops** - Thriving vegetation
        - **Unhealthy Crops** - Stressed or diseased plants
        - **Other** - Additional features
        
        ### 🎯 Features
        - Real-time object detection
        - Image and video processing
        - Detailed statistics and visualizations
        - Adjustable detection parameters
        - Export capabilities
        
        ### 🚀 How to Use
        1. Upload an image or video from drone footage
        2. Adjust detection parameters in the sidebar
        3. View results with bounding boxes and statistics
        4. Download annotated results
        
        ### 📊 Model Information
        - **Architecture**: YOLOv8
        - **Input Size**: 640x640 pixels
        - **Classes**: 4 (Soil, Healthy, Unhealthy, Other)
        - **Framework**: Ultralytics
        
        ### 🛠️ Technical Details
        - Built with Streamlit
        - Powered by Ultralytics YOLO
        - OpenCV for video processing
        - Plotly for visualizations
        
        ### 📝 Credits
        Developed for precision agriculture and crop monitoring using AI-powered drone imagery analysis.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d;'>
        <p>Made with ❤️ for Smart Farming</p>
        <p>© 2024 Farm Detection System</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()