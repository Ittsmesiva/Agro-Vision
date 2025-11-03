import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import time

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Agro Vision - Lemon Detection System",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- EXACT TKINTER GUI STYLING ----------------
st.markdown("""
<style>
    /* Remove default Streamlit padding */
    .main > div {
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    
    /* Hide Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background - dark navy blue */
    .stApp {
        background-color: #0a0e27;
    }
    
    /* Header section - exact match */
    .header-section {
        background: linear-gradient(135deg, #1a1f3a 0%, #0d1129 100%);
        padding: 2rem;
        text-align: center;
        border-bottom: 2px solid #2a3f5f;
        margin-bottom: 0;
    }
    
    .header-emoji {
        font-size: 4rem;
        margin-bottom: -1rem;
    }
    
    .header-title {
        color: #ffd700;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .header-subtitle {
        color: #8899aa;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Main container layout */
    .main-container {
        display: flex;
        gap: 0;
        padding: 0;
        margin: 0;
    }
    
    /* Left panel - Control Panel */
    .control-panel {
        background: #16213e;
        width: 320px;
        padding: 0;
        border-right: 1px solid #2a3f5f;
        min-height: calc(100vh - 160px);
    }
    
    .panel-header {
        background: #1a1f3a;
        padding: 1.5rem;
        border-bottom: 2px solid #ffd700;
    }
    
    .panel-title {
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 0;
    }
    
    .panel-icon {
        margin-right: 0.5rem;
    }
    
    /* Buttons - exact Tkinter style */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 0;
        margin: 0;
        transition: all 0.2s;
    }
    
    /* Button colors matching Tkinter */
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(1) > button {
        background: #2c3e5f;
        color: white;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(1) > button:hover {
        background: #3a4f7a;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(2) > button {
        background: #2c3e5f;
        color: white;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(2) > button:hover {
        background: #3a4f7a;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(3) > button {
        background: #d4145a;
        color: white;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(3) > button:hover {
        background: #e6195f;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(4) > button {
        background: #5a3472;
        color: white;
    }
    
    div[data-testid="column"]:nth-child(1) .stButton:nth-child(4) > button:hover {
        background: #6d4189;
    }
    
    /* Stats section */
    .stats-section {
        background: #16213e;
        padding: 1.5rem;
        border-top: 2px solid #ffd700;
        margin-top: 1rem;
    }
    
    .stats-title {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .stat-item {
        background: #1a1f3a;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-left: 3px solid #667eea;
        border-radius: 4px;
    }
    
    .stat-label {
        color: #8899aa;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
    }
    
    .stat-value {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .stat-value.objects {
        color: #ffd700;
    }
    
    .stat-value.confidence {
        color: #00ccff;
    }
    
    .stat-value.classes {
        color: #ff88cc;
        font-size: 1rem;
    }
    
    /* Right panel - Display */
    .display-panel {
        flex: 1;
        background: #0a0e27;
        padding: 0;
    }
    
    .display-header {
        background: #16213e;
        padding: 1.5rem;
        border-bottom: 2px solid #2a3f5f;
    }
    
    .display-title {
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .display-content {
        background: #0f1419;
        min-height: calc(100vh - 220px);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1.5rem;
        border: 1px solid #2a3f5f;
        border-radius: 4px;
    }
    
    .placeholder {
        text-align: center;
        color: #4a5570;
    }
    
    .placeholder-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }
    
    .placeholder-text {
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .placeholder-subtext {
        font-size: 1rem;
        opacity: 0.7;
    }
    
    /* Footer - exact match */
    .footer-section {
        background: #16213e;
        padding: 1rem 2rem;
        border-top: 1px solid #2a3f5f;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer-status {
        color: #6a7a9a;
        font-size: 0.95rem;
    }
    
    .status-indicator {
        color: #00ff88;
        margin-right: 0.3rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        display: none;
    }
    
    /* Image display */
    .stImage {
        border-radius: 4px;
    }
    
    /* Adjust column spacing */
    div[data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Remove gaps */
    .row-widget {
        gap: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIGURATION ----------------
API_KEY = "mRwKi3IVO4VcXC5ljoix"
MODEL_ID = "agro-vision-lemon-p56kr/2"

# Disease colors (BGR for OpenCV)
DISEASE_COLORS = {
    'Anthracnose': (128, 0, 128),
    'Citrus butterfly': (255, 255, 0),
    'Citrus canker': (255, 0, 255),
    'Citrus Hindu mite': (255, 0, 0),
    'Citrus leaf miner': (0, 128, 255),
    'Healthy': (0, 255, 0),
    'Nutrient Deficiency': (0, 200, 255),
}

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = 0
if 'current_detections' not in st.session_state:
    st.session_state.current_detections = 0
if 'fps' not in st.session_state:
    st.session_state.fps = '--'
if 'confidence' not in st.session_state:
    st.session_state.confidence = '--'
if 'classes' not in st.session_state:
    st.session_state.classes = 'None'
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_resource
def get_client():
    """Cache the Roboflow client"""
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

def draw_predictions(image, predictions):
    """Draw bounding boxes and labels on image"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for pred in predictions:
        class_name = pred['class']
        confidence = pred['confidence']
        x, y = int(pred['x']), int(pred['y'])
        w, h = int(pred['width']), int(pred['height'])
        
        color = DISEASE_COLORS.get(class_name, (0, 165, 255))
        
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        thickness = max(3, int(min(img.shape[0], img.shape[1]) / 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_name} {confidence*100:.0f}%"
        font_scale = max(0.6, min(img.shape[0], img.shape[1]) / 800)
        font_thickness = max(2, int(font_scale * 2))
        
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
        )
        
        label_color = tuple(min(255, c + 80) for c in color)
        padding = int(12 * font_scale)
        cv2.rectangle(img,
                     (x1, y1 - label_h - padding * 2),
                     (x1 + label_w + padding * 2, y1),
                     label_color, -1)
        
        cv2.putText(img, label, (x1 + padding, y1 - padding),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def process_image(image_file, client):
    """Process image and return results"""
    try:
        with open("temp_upload.jpg", "wb") as f:
            f.write(image_file.getvalue())
        
        result = client.infer("temp_upload.jpg", model_id=MODEL_ID)
        return result, None
    except Exception as e:
        return None, str(e)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-section">
    <div class="header-emoji">🍋</div>
    <h1 class="header-title">AGRO VISION</h1>
    <p class="header-subtitle">AI-Powered Lemon Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([320, 1000], gap="small")

# LEFT PANEL - Control Panel
with col1:
    st.markdown("""
    <div class="panel-header">
        <h2 class="panel-title"><span class="panel-icon">⚙</span>Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Select Image File button
    uploaded_file = st.file_uploader(
        "label",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_upload"
    )
    if st.button("📁  Analyze Image", key="select_btn"):
        if uploaded_file:
            st.session_state.current_image = Image.open(uploaded_file)
            st.session_state.webcam_active = False
            
            # Process image
            with st.spinner("Processing..."):
                client = get_client()
                result, error = process_image(uploaded_file, client)
                
                if result:
                    predictions = result.get('predictions', [])
                    st.session_state.current_detections = len(predictions)
                    
                    if predictions:
                        st.session_state.annotated_image = draw_predictions(
                            st.session_state.current_image, predictions
                        )
                        
                        # Update stats
                        avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
                        st.session_state.confidence = f"{avg_conf*100:.1f}%"
                        
                        classes = set(p['class'] for p in predictions)
                        st.session_state.classes = ", ".join(classes)
                    else:
                        st.session_state.annotated_image = st.session_state.current_image
                        st.session_state.confidence = "--"
                        st.session_state.classes = "None"
                    
                    st.session_state.processed_images += 1
                    st.rerun()
                else:
                    st.error(f"Error: {error}")
    
    # Start Webcam button - Toggle webcam
    webcam_btn_text = "⏹  Stop Webcam" if st.session_state.webcam_active else "📷  Start Webcam"
    if st.button(webcam_btn_text, key="webcam_btn"):
        st.session_state.webcam_active = not st.session_state.webcam_active
        if not st.session_state.webcam_active:
            st.session_state.current_image = None
            st.session_state.annotated_image = None
        st.rerun()
    
    # Camera input (shown only when webcam is active)
    if st.session_state.webcam_active:
        camera_photo = st.camera_input("Take a photo", label_visibility="visible", key="camera")
        
        if camera_photo:
            st.session_state.current_image = Image.open(camera_photo)
            
            # Auto-process when photo is taken
            with st.spinner("Analyzing..."):
                client = get_client()
                
                # Convert camera photo
                img_file = io.BytesIO()
                st.session_state.current_image.save(img_file, format='JPEG')
                img_file.seek(0)
                
                # Create a temporary file-like object
                class TempFile:
                    def __init__(self, data):
                        self.data = data
                    def getvalue(self):
                        return self.data
                
                temp_file = TempFile(img_file.getvalue())
                result, error = process_image(temp_file, client)
                
                if result:
                    predictions = result.get('predictions', [])
                    st.session_state.current_detections = len(predictions)
                    
                    if predictions:
                        st.session_state.annotated_image = draw_predictions(
                            st.session_state.current_image, predictions
                        )
                        
                        avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
                        st.session_state.confidence = f"{avg_conf*100:.1f}%"
                        
                        classes = set(p['class'] for p in predictions)
                        st.session_state.classes = ", ".join(classes)
                    else:
                        st.session_state.annotated_image = st.session_state.current_image
                        st.session_state.confidence = "--"
                        st.session_state.classes = "None"
    
    # Save Results button
    if st.button("💾  Save Result", key="save_btn", disabled=st.session_state.annotated_image is None):
        if st.session_state.annotated_image:
            buf = io.BytesIO()
            st.session_state.annotated_image.save(buf, format='JPEG', quality=95)
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="agro_vision_result.jpg",
                mime="image/jpeg",
                key="download_hidden"
            )
    
    # Clear Display button
    if st.button("🗑  Clear Display", key="clear_btn"):
        st.session_state.current_image = None
        st.session_state.annotated_image = None
        st.session_state.current_detections = 0
        st.session_state.confidence = "--"
        st.session_state.classes = "None"
        st.rerun()
    
    # Statistics Section
    st.markdown("""
    <div class="stats-section">
        <div class="stats-title">📊 Detection Statistics</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-item">
        <div class="stat-label">🎯 Objects Detected</div>
        <div class="stat-value objects">{st.session_state.current_detections} Objects</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-item">
        <div class="stat-label">📈 Average Confidence</div>
        <div class="stat-value confidence">{st.session_state.confidence}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-item">
        <div class="stat-label">🏷 Detected Classes</div>
        <div class="stat-value classes">{st.session_state.classes}</div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT PANEL - Display
with col2:
    st.markdown("""
    <div class="display-header">
        <h2 class="display-title">🖼 Detection Display</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display area
    if st.session_state.annotated_image:
        st.image(st.session_state.annotated_image, use_container_width=True)
    else:
        st.markdown("""
        <div class="display-content">
            <div class="placeholder">
                <div class="placeholder-icon">📁</div>
                <div class="placeholder-text">Select an image or start webcam</div>
                <div class="placeholder-subtext">to begin AI-powered detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(f"""
<div class="footer-section">
    <div class="footer-status">
        <span class="status-indicator">🟢</span>System Ready  |  Model: agro-vision-lemon-p56kr/2
    </div>
</div>
""", unsafe_allow_html=True)
