"""
Smart Farm Monitor - AI-Powered Precision Agriculture
Developed by UENR, ATPS & IDRC (2025)
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
        if YOLO is None:
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(str(e))
        return None


def process_image(image, model):
    if model is None:
        st.error("YOLO could not load because OpenCV is unavailable.")
        return None

    img_array = np.array(image)

    # direct tensor inference, avoids cv2
    results = model(img_array, conf=0.25, iou=0.45, verbose=False)
    return results[0]


def draw_detections(image, results):
    """Draw bounding boxes using PIL instead of OpenCV"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        # Draw rectangle (with thicker lines by drawing multiple rectangles)
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
        
        # Prepare label
        label = f"{CLASS_NAMES.get(cls, f'Class {cls}')} {conf:.2f}"
        
        # Get text bounding box
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], fill=color)
        
        # Draw label text
        draw.text((x1 + 5, y1 - text_height - 5), label, fill=(255, 255, 255), font=font)
    
    return img

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
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "ℹ️ About"])
    
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
        
        st.warning("⚠️ Video processing is currently unavailable in this version. Please use the Image Analysis tab for individual frame analysis.")
        
        st.markdown("""
        ### Alternative Approach for Video Analysis:
        1. Extract frames from your video using video editing software
        2. Upload individual frames to the Image Analysis tab
        3. Analyze key frames to monitor crop health over time
        
        **Coming Soon:** Full video processing capabilities will be added in future updates.
        """)
        
        st.info("""
        💡 **Tip:** Focus on analyzing still images at different time points for effective 
        nutrient deficiency monitoring across your cashew farm.
        """)
    
    # Tab 3: About
    with tab3:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 10px; backdrop-filter: blur(5px);">
        <h2>About Smart Farm Monitor</h2>
        
        <h3>🌾 Our Mission</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        The Smart Farm Monitor is an advanced AI-powered system designed specifically for detecting 
        nutrient deficiencies in cashew farms across Africa. Using aerial and drone imagery analysis, 
        this application helps cashew farmers identify nutritional stress early, optimize fertilizer 
        application, and increase crop productivity through precision agriculture techniques.
        </p>
        
        <h3>🎯 Detection Capabilities</h3>
        <ul style='font-size: 1.05rem; line-height: 1.8;'>
            <li><strong>Soil Areas:</strong> Identifying bare ground for monitoring soil exposure and erosion risk</li>
            <li><strong>Healthy Crops:</strong> Recognizing well-nourished cashew plants with optimal nutrient levels</li>
            <li><strong>Unhealthy Crops:</strong> Early detection of nutrient deficiency symptoms in cashew trees</li>
            <li><strong>Other Features:</strong> Additional farm elements and environmental factors</li>
        </ul>
        
        <h3>🌿 Nutrient Deficiency Detection</h3>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
        Nutrient deficiencies in cashew farms often manifest as changes in leaf color, growth patterns, 
        and overall plant vigor. This system analyzes aerial imagery to detect these visual indicators, 
        helping farmers identify which areas of their farm require targeted nutrient supplementation. 
        Early detection prevents yield loss and enables efficient use of fertilizers.
        </p>
        
        <h3>✨ Key Features</h3>
        <ul style='font-size: 1.05rem; line-height: 1.8;'>
            <li>Real-time AI analysis of cashew farm aerial imagery</li>
            <li>Automated detection of nutrient stress indicators</li>
            <li>Support for both images and videos from drones or aircraft</li>
            <li>Detailed statistics and visualizations for farm management decisions</li>
            <li>User-friendly interface accessible to all cashew farmers</li>
            <li>Exportable results for record-keeping and reporting</li>
            <li>Precision detection with confidence metrics</li>
        </ul>
        
        <h3>🤝 Partnership</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        This innovative project is a collaborative initiative supported by:
        </p>
        <ul style='font-size: 1.05rem; line-height: 1.8;'>
            <li><strong>UENR</strong> - University of Energy and Natural Resources</li>
            <li><strong>ATPS</strong> - African Technology Policy Studies Network</li>
            <li><strong>IDRC</strong> - International Development Research Centre</li>
        </ul>
        
        <h3>🎨 Design & Development</h3>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
        This application was designed and developed by the <strong>UENR Research Team</strong> as part 
        of ongoing efforts to bring cutting-edge agricultural technology to African cashew farmers. 
        The system leverages state-of-the-art deep learning algorithms trained specifically on cashew 
        farm imagery to provide accurate and reliable nutrient deficiency detection.
        </p>
        
        <h3>🌍 Impact</h3>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
        By enabling early detection of nutrient deficiencies through aerial imagery analysis, this 
        technology empowers cashew farmers to take targeted corrective actions. This precision approach 
        reduces fertilizer waste, lowers production costs, improves crop yields, and promotes 
        environmentally sustainable farming practices. The system democratizes access to advanced 
        agricultural diagnostics, making them available to smallholder and large-scale farmers alike.
        </p>
        
        <h3>📱 How to Use</h3>
        <ol style='font-size: 1.05rem; line-height: 1.8;'>
            <li>Capture aerial imagery of your cashew farm using a drone</li>
            <li>Upload the image or video to this application</li>
            <li>Wait for the AI system to process and analyze the imagery</li>
            <li>Review the detection results with color-coded indicators</li>
            <li>Examine statistics to understand nutrient status across your farm</li>
            <li>Download annotated results for your records and planning</li>
            <li>Use insights to plan targeted fertilization strategies</li>
        </ol>
        
        <h3>🔬 Technical Approach</h3>
        <p style='font-size: 1.05rem; line-height: 1.8;'>
        The system uses advanced computer vision and machine learning techniques to analyze visual 
        patterns in aerial imagery. By recognizing subtle color variations, growth irregularities, and 
        canopy density patterns, the AI can identify areas where cashew trees are experiencing nutrient 
        stress before symptoms become severe enough to significantly impact yields.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
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
