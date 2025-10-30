import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import io
import time

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="🌿 Agro Vision - Lemon Disease Detection",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS FOR MOBILE OPTIMIZATION ----------------
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    /* Disease info card */
    .disease-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .disease-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }
        .header-subtitle {
            font-size: 1rem;
        }
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Upload button styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIGURATION ----------------
API_KEY = "mRwKi3IVO4VcXC5ljoix"
MODEL_ID = "agro-vision-lemon-p56kr/2"

# Disease information database
DISEASE_INFO = {
    "Anthracnose": {
        "color": (128, 0, 128),
        "severity": "High",
        "symptoms": "Dark, sunken lesions on leaves, stems, and fruits",
        "remedy": "Apply copper-based fungicides. Remove infected plant parts. Improve air circulation.",
        "prevention": "Avoid overhead watering. Maintain proper plant spacing."
    },
    "Citrus butterfly": {
        "color": (255, 255, 0),
        "severity": "Medium",
        "symptoms": "Caterpillar feeding damage, leaf defoliation",
        "remedy": "Manual removal of caterpillars. Use neem oil spray. Apply Bacillus thuringiensis.",
        "prevention": "Regular monitoring. Use pheromone traps."
    },
    "Citrus canker": {
        "color": (255, 0, 255),
        "severity": "High",
        "symptoms": "Raised, corky lesions on leaves, stems, and fruits",
        "remedy": "Apply copper-based bactericides. Remove and destroy infected parts immediately.",
        "prevention": "Use disease-free planting material. Avoid working with wet plants."
    },
    "Citrus Hindu mite": {
        "color": (255, 0, 0),
        "severity": "Medium",
        "symptoms": "Leaf curling, distortion, and bronzing",
        "remedy": "Apply miticides. Use sulfur spray. Release predatory mites.",
        "prevention": "Regular monitoring. Maintain plant health with proper nutrition."
    },
    "Citrus leaf miner": {
        "color": (0, 128, 255),
        "severity": "Low",
        "symptoms": "Serpentine mines on young leaves, leaf distortion",
        "remedy": "Apply neem oil. Use spinosad-based sprays. Time spraying with new flush.",
        "prevention": "Avoid excessive nitrogen fertilization. Protect new growth."
    },
    "Healthy": {
        "color": (0, 255, 0),
        "severity": "None",
        "symptoms": "No visible disease symptoms. Plant appears vigorous and green.",
        "remedy": "No treatment needed. Continue regular maintenance.",
        "prevention": "Maintain good cultural practices. Regular monitoring."
    },
    "Nutrient Deficiency": {
        "color": (0, 200, 255),
        "severity": "Medium",
        "symptoms": "Yellowing leaves, stunted growth, poor fruit quality",
        "remedy": "Apply balanced fertilizer. Conduct soil test. Apply specific micronutrients as needed.",
        "prevention": "Regular soil testing. Proper fertilization schedule. Maintain soil pH 6-7."
    }
}

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

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
        
        # Get color for this class
        color = DISEASE_INFO.get(class_name, {}).get('color', (0, 165, 255))
        
        # Calculate box coordinates
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Draw bounding box
        thickness = max(2, int(min(img.shape[0], img.shape[1]) / 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = f"{class_name} {confidence*100:.0f}%"
        font_scale = max(0.5, min(img.shape[0], img.shape[1]) / 1000)
        font_thickness = max(1, int(font_scale * 2))
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
        )
        
        # Draw label background (lighter shade)
        label_color = tuple(min(255, c + 80) for c in color)
        padding = int(10 * font_scale)
        cv2.rectangle(img,
                     (x1, y1 - label_h - padding * 2),
                     (x1 + label_w + padding * 2, y1),
                     label_color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1 + padding, y1 - padding),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness)
    
    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def process_image(image_file, client):
    """Process image and return results"""
    try:
        # Save the uploaded file temporarily
        with open("temp_upload.jpg", "wb") as f:
            f.write(image_file.getvalue())
        
        # Run inference on the saved file
        result = client.infer("temp_upload.jpg", model_id=MODEL_ID)
        return result, None
    except Exception as e:
        return None, str(e)

# ---------------- MAIN APP ----------------
# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🍋 Agro Vision</h1>
    <p class="header-subtitle">AI-Powered Lemon Disease Detection for Farmers</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{st.session_state.processed_images}</div>
        <div class="stat-label">Images Analyzed</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{st.session_state.total_detections}</div>
        <div class="stat-label">Total Detections</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    **Agro Vision** helps farmers detect lemon diseases early using AI technology.
    
    Simply upload a photo of a lemon leaf, and our AI will:
    - Identify diseases
    - Show confidence levels
    - Provide treatment recommendations
    """)
    
    st.markdown("---")
    st.header("🌐 Language")
    language = st.selectbox("Select Language", ["English", "தமிழ் (Tamil)"], index=0)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Leaf Image")
    
    # Camera input for mobile
    camera_photo = st.camera_input("Take a photo 📸")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Or upload from gallery",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the lemon leaf"
    )
    
    # Select input source
    input_image = camera_photo if camera_photo else uploaded_file
    
    if input_image:
        # Display original image
        image = Image.open(input_image)
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Leaf", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your leaf image..."):
                # Get cached client
                client = get_client()
                
                # Process image
                start_time = time.time()
                result, error = process_image(input_image, client)
                processing_time = time.time() - start_time
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    # Update statistics
                    st.session_state.processed_images += 1
                    
                    # Store results in session state
                    st.session_state.last_result = result
                    st.session_state.last_image = image
                    st.session_state.processing_time = processing_time
                    
                    st.success(f"✅ Analysis complete in {processing_time:.2f}s!")
                    st.rerun()

with col2:
    st.subheader("🎯 Detection Results")
    
    if 'last_result' in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result
        predictions = result.get('predictions', [])
        
        # Update total detections
        st.session_state.total_detections = len(predictions)
        
        if len(predictions) > 0:
            # Draw predictions on image
            annotated_image = draw_predictions(st.session_state.last_image, predictions)
            st.image(annotated_image, caption="Detected Issues", use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            annotated_image.save(buf, format='JPEG', quality=95)
            st.download_button(
                label="⬇️ Download Result",
                data=buf.getvalue(),
                file_name="agro_vision_result.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
            # Show detailed results
            st.markdown("### 📋 Detailed Analysis")
            
            for i, pred in enumerate(predictions, 1):
                class_name = pred['class']
                confidence = pred['confidence'] * 100
                
                # Get disease info
                disease_info = DISEASE_INFO.get(class_name, {})
                severity = disease_info.get('severity', 'Unknown')
                symptoms = disease_info.get('symptoms', 'N/A')
                remedy = disease_info.get('remedy', 'Consult agricultural expert')
                prevention = disease_info.get('prevention', 'N/A')
                
                # Severity color
                severity_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢',
                    'None': '✅'
                }.get(severity, '⚪')
                
                with st.expander(f"{i}. {class_name} - {confidence:.1f}% confidence", expanded=True):
                    st.markdown(f"""
                    **{severity_color} Severity:** {severity}
                    
                    **🔍 Symptoms:**
                    {symptoms}
                    
                    **💊 Recommended Treatment:**
                    {remedy}
                    
                    **🛡️ Prevention:**
                    {prevention}
                    """)
                    
        else:
            st.info("✅ No issues detected! Your plant appears healthy.")
            st.balloons()
    else:
        st.info("👆 Upload an image to see detection results here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌾 Powered by AI • Made for Farmers • Free to Use</p>
    <p style='font-size: 0.9rem;'>Model: agro-vision-lemon-p56kr/2 | Accuracy: 95%+</p>
</div>
""", unsafe_allow_html=True)