"""
Smart Farm Monitor - AI-Powered Precision Agriculture
Developed by UENR, ATPS & IDRC (2025)
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
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
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Class information
CLASS_NAMES = {0: 'Soil', 1: 'Healthy Crop', 2: 'Unhealthy Crop', 3: 'Other'}
CLASS_COLORS = {0: (139, 69, 19), 1: (34, 139, 34), 2: (255, 69, 0), 3: (128, 128, 128)}
CLASS_COLORS_HEX = {0: '#8B4513', 1: '#228B22', 2: '#FF4500', 3: '#808080'}

@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    img_array = np.array(image)
    results = model.predict(img_array, conf=0.25, iou=0.45, verbose=False)
    return results[0]

def draw_detections(image, results):
    img_array = np.array(image)
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        label = f"{CLASS_NAMES.get(cls, f'Class {cls}')} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_array, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
        cv2.putText(img_array, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return Image.fromarray(img_array)

def get_detection_stats(results):
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    class_counts = {}
    for cls in classes:
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    class_conf = {}
    for cls in set(classes):
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        cls_confidences = confidences[classes == cls]
        class_conf[class_name] = np.mean(cls_confidences)
    
    return class_counts, class_conf

def main():
    # Get background image as base64
    bg_image = get_base64_image("aerial1.jpeg")
    
    # Custom CSS with background image
    background_style = ""
    if bg_image:
        background_style = f"""
        .stApp {{
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)),
                url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    
    st.markdown(f"""
        <style>
        {background_style}
        
        .main-header {{
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(255,255,255,0.9);
            font-weight: bold;
        }}
        .sub-header {{
            font-size: 1.2rem;
            color: #1a1a1a;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 600;
            text-shadow: 1px 1px 3px rgba(255,255,255,0.9);
        }}
        
        /* Make text more readable on faded background */
        .stMarkdown, p, li, span {{
            text-shadow: 0px 0px 2px rgba(255,255,255,0.8);
        }}
        
        /* File uploader */
        div[data-testid="stFileUploader"] {{
            background-color: rgba(255, 255, 255, 0.7);
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px dashed #4caf50;
            backdrop-filter: blur(3px);
        }}
        
        /* Info boxes */
        div[data-testid="stAlert"] {{
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(3px);
        }}
        
        .detection-box {{
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(240, 242, 246, 0.9);
            margin: 10px 0;
            backdrop-filter: blur(5px);
        }}
        
        .stats-section {{
            background-color: rgba(232, 245, 233, 0.85);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
            margin: 20px 0;
            backdrop-filter: blur(3px);
        }}
        
        .stSidebar {{
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }}
        
        div[data-testid="stExpander"] {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 8px;
            backdrop-filter: blur(3px);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.9);
            padding: 0.5rem;
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 3rem;
            background: transparent;
            border-radius: 8px;
            padding: 0 1.5rem;
            font-weight: 600;
            color: #374151;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }}
        
        /* Buttons */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            transition: all 0.2s;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }}
        
        /* Download button */
        .stDownloadButton>button {{
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: #059669;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-weight: 600;
            color: #374151;
        }}
        
        /* Images */
        img {{
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        /* Video */
        video {{
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        /* Hide streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Smart Farm Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nutrient Deficiency Detection in Cashew Farms Using Aerial Imagery</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This application analyzes drone and aerial imagery to detect nutrient 
            deficiencies in cashew farms using advanced AI technology.
            
            **Detection Classes:**
            - 🟤 Soil (Bare ground areas)
            - 🟢 Healthy Crops (Thriving vegetation)
            - 🔴 Unhealthy Crops (Nutrient deficient plants)
            - ⚪ Other (Additional features)
            """
        )
        
        st.header("Instructions")
        st.markdown("""
        1. Upload aerial/drone cashew farm imagery
        2. Wait for AI analysis
        3. View detection results with bounding boxes
        4. Review statistics and metrics
        5. Download annotated results for your records
        """)
        
        st.header("📋 Detection Classes")
        for cls_id, cls_name in CLASS_NAMES.items():
            color = CLASS_COLORS_HEX[cls_id]
            st.markdown(
                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; '
                f'border-radius: 4px; margin-right: 10px;"></div>'
                f'<span style="font-weight: 600;">{cls_name}</span></div>',
                unsafe_allow_html=True
            )
    
    # Load model
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        st.error("🚨 Model not found. Please ensure the model is at: farm_detection/yolov8n_farm/weights/best.pt")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Tabs
    tab1, tab2= st.tabs(["📷 Image Analysis", "🎥 Video Analysis"])
    
    # Tab 1: Image Analysis
    with tab1:
        st.subheader("📤 Upload Aerial/Drone Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload aerial or drone imagery of your cashew farm"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
            
            with col2:
                st.subheader("🔍 Detection Results")
                
                with st.spinner("Analyzing aerial imagery..."):
                    try:
                        results = process_image(image, model)
                        annotated_image = draw_detections(image, results)
                        
                        st.image(annotated_image, use_container_width=True)
                        
                        # Get detection count
                        num_detections = len(results.boxes)
                        st.caption(f"✅ Detected {num_detections} objects")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
            
            # Statistics Section
            st.markdown("---")
            st.subheader("📊 Detection Statistics")
            
            class_counts, class_conf = get_detection_stats(results)
            
            if class_counts:
                # Metrics row
                cols = st.columns(len(class_counts))
                for idx, (class_name, count) in enumerate(class_counts.items()):
                    with cols[idx]:
                        st.metric(
                            label=class_name,
                            value=f"{count}",
                            delta=f"{class_conf.get(class_name, 0)*100:.1f}% conf"
                        )
                
                # Visualization
                st.markdown("---")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Detection Counts**")
                    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
                    fig = px.bar(df, x='Class', y='Count', color='Class',
                               color_discrete_map={'Soil': '#8B4513', 'Healthy Crop': '#228B22',
                                                 'Unhealthy Crop': '#FF4500', 'Other': '#808080'})
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    st.markdown("**Average Confidence**")
                    df_conf = pd.DataFrame(list(class_conf.items()), columns=['Class', 'Confidence'])
                    df_conf['Confidence'] = df_conf['Confidence'] * 100
                    fig_conf = px.bar(df_conf, x='Class', y='Confidence', color='Class',
                                    color_discrete_map={'Soil': '#8B4513', 'Healthy Crop': '#228B22',
                                                      'Unhealthy Crop': '#FF4500', 'Other': '#808080'})
                    fig_conf.update_layout(showlegend=False, height=300, yaxis_range=[0, 100])
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Download button
                st.markdown("---")
                img_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                annotated_image.save(img_bytes.name)
                with open(img_bytes.name, 'rb') as f:
                    st.download_button(
                        "📥 Download Annotated Image",
                        f,
                        "farm_analysis_result.png",
                        "image/png"
                    )
            else:
                st.info("🔍 No objects detected. Try adjusting the image or using a different aerial view.")
        
        else:
            st.info("👆 Please upload an aerial or drone image of your cashew farm to get started")
            
            # Information cards
            st.markdown("---")
            st.subheader("📚 How This System Helps Cashew Farmers")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="detection-box">
                <h4>✅ Early Nutrient Detection</h4>
                <p>Identify nutrient deficiencies in cashew trees before they severely impact yield and quality</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="detection-box">
                <h4>📊 Precision Agriculture</h4>
                <p>Make data-driven decisions about fertilizer application and nutrient management strategies</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="detection-box">
                <h4>💰 Cost Optimization</h4>
                <p>Apply fertilizers only where needed, reducing costs and environmental impact</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 2: Video Analysis
    with tab2:
        st.subheader("🎬 Upload Aerial/Drone Video")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload aerial or drone video of your cashew farm",
            key="video"
        )
        
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Video**")
                st.video(tfile.name)
            
            if st.button("🚀 Analyze Video"):
                st.markdown("### ⚙️ Processing Video...")
                progress = st.progress(0)
                
                with st.spinner("AI is analyzing your video..."):
                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    frame_count = 0
                    class_counts = {}
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model.predict(frame, conf=0.25, iou=0.45, verbose=False)[0]
                        annotated_frame = results.plot()
                        
                        for cls in results.boxes.cls.cpu().numpy().astype(int):
                            class_name = CLASS_NAMES.get(cls, f"Class {cls}")
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        out.write(annotated_frame)
                        frame_count += 1
                        progress.progress(frame_count / total_frames)
                    
                    cap.release()
                    out.release()
                
                st.success("✅ Video analysis completed!")
                
                with col2:
                    st.markdown("**Analyzed Video**")
                    st.video(output_path)
                
                # Video statistics
                if class_counts:
                    st.markdown("---")
                    st.subheader("📊 Video Detection Statistics")
                    
                    cols = st.columns(len(class_counts))
                    for idx, (class_name, count) in enumerate(class_counts.items()):
                        with cols[idx]:
                            st.metric(label=class_name, value=f"{count} total")
                
                # Download
                with open(output_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Analyzed Video",
                        f,
                        "farm_video_analysis.mp4",
                        "video/mp4"
                    )
    
    
    
    # Footer - Developer Credits at the bottom
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; background-color: rgba(255, 255, 255, 0.9); 
                    border-radius: 10px; backdrop-filter: blur(5px); margin-top: 2rem;'>
            <p style='font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem;'>
                🌾 Smart Farm Monitor
            </p>
            <p style='font-size: 1rem; color: #374151; margin-bottom: 0.5rem;'>
                <strong>Developed by:</strong>
            </p>
            <p style='font-size: 1.1rem; font-weight: 600; color: #059669; margin-bottom: 0.5rem;'>
                UENR | ATPS | IDRC
            </p>
            <p style='font-size: 0.95rem; color: #6b7280; margin-top: 1rem;'>
                © 2025 | Empowering Cashew Farmers through AI Technology
            </p>
            <p style='font-size: 0.9rem; color: #9ca3af; margin-top: 0.5rem;'>
                <em>For inquiries and support, contact the UENR Research Team</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
