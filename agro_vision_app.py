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

# Comprehensive Disease Information Database
DISEASE_INFO = {
    'Anthracnose': {
        'severity': '🔴 High Risk',
        'description': 'Fungal disease causing dark, sunken lesions on leaves, stems, and fruits. Can lead to severe defoliation and fruit drop.',
        'symptoms': [
            'Dark brown to black circular spots on leaves',
            'Sunken lesions with raised borders',
            'Premature fruit drop',
            'Twig dieback in severe cases',
            'Pink spore masses in humid conditions'
        ],
        'immediate_cure': [
            '1. Remove and destroy all infected plant parts immediately',
            '2. Apply copper oxychloride (0.3%) or mancozeb (0.25%) spray',
            '3. Spray with carbendazim (0.1%) for systemic control',
            '4. Repeat application every 10-15 days',
            '5. Ensure proper drainage around plants'
        ],
        'long_term_treatment': [
            'Apply preventive copper sprays before rainy season',
            'Prune to improve air circulation',
            'Use resistant varieties when replanting',
            'Maintain balanced fertilization (avoid excess nitrogen)',
            'Apply mulch to prevent soil splash on leaves'
        ],
        'prevention': [
            'Plant in well-drained soil with good air circulation',
            'Avoid overhead irrigation',
            'Remove fallen leaves and fruits regularly',
            'Prune trees to allow sunlight penetration',
            'Apply preventive fungicide sprays during monsoon',
            'Use drip irrigation instead of sprinklers'
        ],
        'organic_remedies': [
            'Neem oil spray (5ml/liter) weekly',
            'Bordeaux mixture (1%) before flowering',
            'Garlic extract spray as natural fungicide',
            'Trichoderma viride application to soil'
        ]
    },
    'Citrus butterfly': {
        'severity': '🟡 Medium Risk',
        'description': 'Caterpillar pest that feeds on young citrus leaves, causing defoliation and stunted growth.',
        'symptoms': [
            'Chewed and damaged young leaves',
            'Presence of green/brown caterpillars on plants',
            'Defoliation of new shoots',
            'Bird dropping-like caterpillars',
            'Irregular holes in leaves'
        ],
        'immediate_cure': [
            '1. Handpick and destroy caterpillars manually',
            '2. Spray Bacillus thuringiensis (Bt) at 1g/liter',
            '3. Apply neem-based insecticide (5ml/liter)',
            '4. Use spinosad spray for heavy infestation',
            '5. Repeat every 7-10 days if needed'
        ],
        'long_term_treatment': [
            'Install pheromone traps to monitor butterfly population',
            'Encourage natural predators (birds, wasps)',
            'Regular monitoring of new growth',
            'Apply neem cake to soil (200g per plant)',
            'Maintain plant health with balanced nutrition'
        ],
        'prevention': [
            'Protect young plants with fine mesh netting',
            'Plant companion plants like marigold',
            'Regular inspection of new shoots',
            'Avoid excessive nitrogen fertilization',
            'Maintain garden hygiene',
            'Use sticky traps for adult butterflies'
        ],
        'organic_remedies': [
            'Neem oil spray (3ml/liter) weekly',
            'Garlic-chili spray as repellent',
            'Soap solution spray (5ml liquid soap/liter)',
            'Encourage beneficial insects like ladybugs'
        ]
    },
    'Citrus canker': {
        'severity': '🔴 High Risk - Bacterial',
        'description': 'Highly contagious bacterial disease causing raised corky lesions. Can spread rapidly and severely damage crops.',
        'symptoms': [
            'Raised, corky lesions with yellow halos',
            'Brown spots on leaves, stems, and fruits',
            'Premature leaf and fruit drop',
            'Fruit blemishes reducing market value',
            'Lesions on both sides of leaves'
        ],
        'immediate_cure': [
            '1. ISOLATE infected plants immediately',
            '2. Remove and burn all infected parts',
            '3. Spray copper hydroxide (0.3%) thoroughly',
            '4. Apply streptocycline (100 ppm) + copper oxychloride',
            '5. Disinfect pruning tools with bleach solution',
            '6. Repeat sprays every 7-10 days'
        ],
        'long_term_treatment': [
            'Quarantine new plants for 30 days before planting',
            'Use only disease-free planting material',
            'Apply copper sprays preventively during monsoon',
            'Maintain windbreaks to reduce wind-blown spread',
            'Remove severely infected trees completely'
        ],
        'prevention': [
            'CRITICAL: Plant only certified disease-free nursery stock',
            'Never work with plants when wet',
            'Implement strict quarantine protocols',
            'Control citrus leaf miner (vector)',
            'Use copper sprays before rain',
            'Disinfect all tools and equipment',
            'Avoid overhead irrigation',
            'Plant windbreaks to reduce bacterial spread'
        ],
        'organic_remedies': [
            'Bordeaux mixture (1%) spray',
            'Copper-based organic fungicides',
            'Maintain plant vigor with compost',
            'Remove volunteer citrus plants nearby'
        ],
        'legal_note': 'Citrus canker is a quarantine disease in many regions. Report to local agricultural authorities.'
    },
    'Citrus Hindu mite': {
        'severity': '🟡 Medium Risk',
        'description': 'Microscopic mites causing leaf curling, distortion, and bronzing. Thrive in hot, dry conditions.',
        'symptoms': [
            'Downward curling of leaf edges',
            'Silvering or bronzing of leaves',
            'Leaf distortion and stunting',
            'Fruit skin russeting',
            'Fine webbing under leaves (severe cases)'
        ],
        'immediate_cure': [
            '1. Spray wettable sulfur (2g/liter) thoroughly',
            '2. Apply abamectin or spiromesifen for heavy infestation',
            '3. Use horticultural oil (10ml/liter)',
            '4. Ensure spray reaches underside of leaves',
            '5. Repeat every 7-10 days, alternate chemicals'
        ],
        'long_term_treatment': [
            'Release predatory mites (Amblyseius species)',
            'Maintain regular sulfur dust application',
            'Improve irrigation to increase humidity',
            'Monitor with 10X hand lens regularly',
            'Apply neem oil (5ml/liter) as suppression'
        ],
        'prevention': [
            'Avoid water stress - maintain consistent moisture',
            'Increase humidity around plants (mist occasionally)',
            'Remove heavily infested leaves',
            'Encourage natural predators',
            'Avoid excessive dust on leaves',
            'Plant windbreaks to reduce mite migration',
            'Use reflective mulches'
        ],
        'organic_remedies': [
            'Sulfur dust application weekly',
            'Neem oil spray (5ml/liter)',
            'Garlic-chili spray as repellent',
            'Release predatory mites',
            'Spray leaves with water to reduce populations'
        ]
    },
    'Citrus leaf miner': {
        'severity': '🟢 Low to Medium Risk',
        'description': 'Moth larvae tunnel through young leaves creating serpentine mines. Mainly affects new growth.',
        'symptoms': [
            'Silvery serpentine trails on leaves',
            'Leaf curling and distortion',
            'Stunted new growth',
            'Secondary infections in damaged tissue',
            'Reduced photosynthesis'
        ],
        'immediate_cure': [
            '1. Spray spinosad (0.5ml/liter) on new flush',
            '2. Apply imidacloprid soil drench (0.3ml/liter)',
            '3. Use neem oil (5ml/liter) on affected plants',
            '4. Remove heavily mined leaves',
            '5. Time sprays with new leaf emergence'
        ],
        'long_term_treatment': [
            'Install pheromone traps for monitoring',
            'Time fertilization to avoid excessive flushing',
            'Encourage natural parasitoid wasps',
            'Apply systemic insecticides during growth periods',
            'Prune to synchronize new growth'
        ],
        'prevention': [
            'Time pruning to avoid continuous flushing',
            'Avoid excessive nitrogen fertilization',
            'Use insect-proof netting on young plants',
            'Protect new growth with preventive sprays',
            'Encourage natural enemies (wasps, beetles)',
            'Apply sticky traps to catch adult moths',
            'Maintain balanced plant nutrition'
        ],
        'organic_remedies': [
            'Neem oil spray (3-5ml/liter) on new flush',
            'Azadirachtin-based products',
            'Spinosad (organic approved)',
            'Release Ageniaspis citricola (parasitoid wasp)',
            'Spray kaolin clay as physical barrier'
        ]
    },
    'Healthy': {
        'severity': '✅ No Risk - Plant is Healthy',
        'description': 'Plant shows no signs of disease or pest damage. Maintain current care practices.',
        'symptoms': [
            'Vibrant green foliage',
            'No spots, lesions, or discoloration',
            'Normal growth and development',
            'Good fruit set and quality',
            'No pest presence'
        ],
        'immediate_cure': [
            'No treatment needed',
            'Continue current maintenance practices',
            'Monitor regularly for any changes'
        ],
        'long_term_treatment': [
            'Maintain balanced fertilization schedule',
            'Continue proper watering practices',
            'Regular monitoring for early problem detection',
            'Maintain soil health with organic matter',
            'Prune for good air circulation'
        ],
        'prevention': [
            'Continue regular monitoring (weekly)',
            'Maintain balanced NPK fertilization',
            'Ensure proper drainage',
            'Prune to maintain air circulation',
            'Practice crop rotation if possible',
            'Use mulch to conserve moisture',
            'Keep area free of weeds and debris',
            'Apply preventive sprays during disease-prone seasons'
        ],
        'organic_remedies': [
            'Compost tea application monthly',
            'Neem cake soil amendment',
            'Seaweed extract foliar spray',
            'Maintain beneficial insect populations'
        ]
    },
    'Nutrient Deficiency': {
        'severity': '🟡 Medium Risk - Requires Attention',
        'description': 'Insufficient essential nutrients affecting plant health, growth, and productivity.',
        'symptoms': [
            'Yellowing of leaves (chlorosis)',
            'Stunted growth and small leaves',
            'Poor fruit quality and yield',
            'Leaf necrosis or browning',
            'Weak stems and branches'
        ],
        'immediate_cure': [
            '1. Conduct soil test to identify specific deficiency',
            '2. Apply foliar spray of micronutrients (Zn, Fe, Mn, B)',
            '3. Apply balanced NPK fertilizer (19:19:19) at 50g per plant',
            '4. For quick response: Use water-soluble fertilizers',
            '5. Correct soil pH if needed (ideal 6.0-7.0)'
        ],
        'long_term_treatment': [
            'Nitrogen deficiency: Apply urea or ammonium sulfate',
            'Phosphorus deficiency: Apply superphosphate',
            'Potassium deficiency: Apply potassium sulfate',
            'Iron deficiency: Apply chelated iron or ferrous sulfate',
            'Zinc deficiency: Apply zinc sulfate (0.5%)',
            'Magnesium deficiency: Apply Epsom salt (1%)',
            'Boron deficiency: Apply borax (carefully)',
            'Regular fertilization schedule (every 2-3 months)'
        ],
        'prevention': [
            'Annual soil testing',
            'Follow recommended fertilization schedule',
            'Apply organic matter (compost, manure) annually',
            'Maintain proper soil pH (6.0-7.0)',
            'Use balanced fertilizers',
            'Apply micronutrients based on soil test',
            'Improve soil structure with organic amendments',
            'Avoid over-watering (reduces nutrient uptake)'
        ],
        'organic_remedies': [
            'Vermicompost application (2-3 kg per plant)',
            'Compost tea foliar spray',
            'Seaweed extract (rich in micronutrients)',
            'Bone meal for phosphorus',
            'Wood ash for potassium',
            'Epsom salt for magnesium',
            'Fish emulsion for nitrogen',
            'Green manure incorporation'
        ],
        'fertilization_schedule': {
            'Young plants (1-3 years)': '50-100g NPK every 2 months',
            'Bearing plants (4+ years)': '200-500g NPK every 3 months',
            'Micronutrients': 'Spray every 3-4 months',
            'Organic matter': '10-20 kg compost annually'
        }
    }
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
        <h2 class="panel-title"><span class="panel-icon">⚙️</span>Control Panel</h2>
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
    if st.button("📁  Select Image File", key="select_btn"):
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
                    st.session_state.last_predictions = predictions  # Store for disease info display
                    
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
    
    # Start Webcam button
    camera_photo = st.camera_input("label", label_visibility="collapsed", key="camera")
    if st.button("📷  Start Webcam", key="webcam_btn"):
        if camera_photo:
            st.session_state.current_image = Image.open(camera_photo)
            st.session_state.webcam_active = True
            
            # Process image
            with st.spinner("Processing..."):
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
                    st.session_state.last_predictions = predictions  # Store for disease info display
                    
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
                    
                    st.rerun()
    
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
    if st.button("🗑️  Clear Display", key="clear_btn"):
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
        <div class="stat-label">⚡ Frame Rate</div>
        <div class="stat-value">{st.session_state.fps} FPS</div>
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
        <div class="stat-label">🏷️ Detected Classes</div>
        <div class="stat-value classes">{st.session_state.classes}</div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT PANEL - Display
with col2:
    st.markdown("""
    <div class="display-header">
        <h2 class="display-title">🖼️ Detection Display</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display area
    if st.session_state.annotated_image:
        st.image(st.session_state.annotated_image, use_container_width=True)
        
        # Show detailed disease information
        if st.session_state.current_detections > 0:
            st.markdown("---")
            st.markdown("### 📋 Detailed Disease Information & Treatment")
            
            # Get unique detected classes
            if 'last_predictions' in st.session_state:
                detected_classes = set(p['class'] for p in st.session_state.last_predictions)
                
                for disease in detected_classes:
                    if disease in DISEASE_INFO:
                        info = DISEASE_INFO[disease]
                        
                        with st.expander(f"🌿 {disease} - {info['severity']}", expanded=True):
                            # Description
                            st.markdown(f"**📖 Description:**")
                            st.info(info['description'])
                            
                            # Symptoms
                            st.markdown("**🔍 Symptoms to Look For:**")
                            for symptom in info['symptoms']:
                                st.markdown(f"- {symptom}")
                            
                            st.markdown("---")
                            
                            # Immediate Cure
                            st.markdown("**💊 IMMEDIATE TREATMENT (Start Now):**")
                            st.error("⚠️ Take action within 24-48 hours")
                            for step in info['immediate_cure']:
                                st.markdown(f"**{step}**")
                            
                            st.markdown("---")
                            
                            # Long-term Treatment
                            st.markdown("**🔬 LONG-TERM MANAGEMENT (Next 2-4 weeks):**")
                            for treatment in info['long_term_treatment']:
                                st.markdown(f"- {treatment}")
                            
                            st.markdown("---")
                            
                            # Prevention
                            st.markdown("**🛡️ PREVENTION (Future Protection):**")
                            st.success("Follow these practices to avoid recurrence:")
                            for prevention in info['prevention']:
                                st.markdown(f"✓ {prevention}")
                            
                            st.markdown("---")
                            
                            # Organic Remedies
                            st.markdown("**🌱 ORGANIC/NATURAL REMEDIES:**")
                            for remedy in info['organic_remedies']:
                                st.markdown(f"🌿 {remedy}")
                            
                            # Special notes
                            if 'legal_note' in info:
                                st.warning(f"⚠️ **IMPORTANT:** {info['legal_note']}")
                            
                            if 'fertilization_schedule' in info:
                                st.markdown("---")
                                st.markdown("**📅 RECOMMENDED FERTILIZATION SCHEDULE:**")
                                for timing, amount in info['fertilization_schedule'].items():
                                    st.markdown(f"- **{timing}:** {amount}")
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
