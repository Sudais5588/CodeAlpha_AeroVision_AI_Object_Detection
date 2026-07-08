import os
import cv2
import av
import time
import tempfile
import threading
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from html import escape
from streamlit.components.v1 import html as st_html

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AeroVision AI | Object Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# SESSION STATE
# =========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

if "language_key" not in st.session_state:
    st.session_state.language_key = "EN"

if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if "menu_open" not in st.session_state:
    st.session_state.menu_open = False

if "active_menu_item" not in st.session_state:
    st.session_state.active_menu_item = None

if "menu_loading_item" not in st.session_state:
    st.session_state.menu_loading_item = None


# =========================================================
# LANGUAGE DATA
# =========================================================
LANGUAGE_NAMES = {
    "EN": "English",
    "RU": "Roman Urdu",
    "UR": "Urdu"
}

TEXT = {
    "EN": {
        "hero_title": "AeroVision AI Object Detection",
        "hero_subtitle": "AI-powered object detection and tracking dashboard for images, videos, and live camera streams.",
        "overview": "This platform uses a YOLO vision model to detect and track objects from multiple input sources. Upload an image, process a video, or start your webcam to perform real-time detection with confidence and resolution control.",
        "products": "Products",
        "services": "Services",
        "about": "About",
        "process": "Process",
        "contact": "Contact",
        "menu_services": "Image detection, video detection, object tracking, and live camera detection.",
        "menu_about": "A professional AI vision platform built with Streamlit, YOLO, and OpenCV.",
        "menu_process": "Upload input, choose confidence and resolution, run detection, and download results.",
        "menu_contact": "Developer Contact: dev@aerovision.ai",
        "control": "Control Center",
        "model": "Select YOLO Model",
        "confidence": "Detection Confidence",
        "resolution": "Inference Resolution",
        "tracking": "Enable Object Tracking",
        "summary": "Show Detected Class Summary",
        "tip": "Use yolov8n.pt for faster performance. Use yolov8s.pt for better accuracy but slower speed.",
        "execution": "Execution Contexts",
        "image_card": "Image Detection Dashboard",
        "image_desc": "Upload JPG, JPEG, or PNG images and detect objects instantly.",
        "video_card": "Video Detection Dashboard",
        "video_desc": "Upload MP4, AVI, or MOV videos and process detection frame by frame.",
        "live_card": "Live Detection Engine",
        "live_desc": "Use your webcam for real-time object detection and tracking.",
        "image_tab": "Image Detection",
        "video_tab": "Video Detection",
        "live_tab": "Live Detection",
        "upload_image": "Upload an image",
        "upload_video": "Upload a video",
        "run_image": "Run Image Detection",
        "run_video": "Run Video Detection",
        "original": "Original Image",
        "result": "Detection Result",
        "objects": "Objects Detected",
        "classes": "Unique Classes",
        "conf_level": "Confidence Level",
        "detected_classes": "Detected Classes",
        "image_info": "Upload an image to start object detection.",
        "video_info": "Upload a video to start video detection.",
        "live_info": "Start your webcam below. For best performance, use yolov8n.pt and medium resolution.",
        "processing": "Processing video. Please wait...",
        "processed": "Processed Detection Video",
        "download": "Download Processed Video",
        "footer": "AeroVision AI · Object Detection & Tracking Tool · Built with Streamlit, YOLO and OpenCV"
    },
    "RU": {
        "hero_title": "AeroVision AI Object Detection",
        "hero_subtitle": "Images, videos aur live camera streams ke liye AI detection aur tracking dashboard.",
        "overview": "Ye platform YOLO vision model use karta hai jo images, videos aur webcam se objects detect aur track karta hai. Aap confidence aur resolution control set kar sakte hain.",
        "products": "Products",
        "services": "Services",
        "about": "About",
        "process": "Process",
        "contact": "Contact",
        "menu_services": "Image detection, video detection, object tracking aur live camera detection.",
        "menu_about": "Streamlit, YOLO aur OpenCV se bana hua professional AI vision platform.",
        "menu_process": "Input upload karein, confidence aur resolution choose karein, detection run karein, phir result download karein.",
        "menu_contact": "Developer Contact: dev@aerovision.ai",
        "control": "Control Center",
        "model": "YOLO Model select karein",
        "confidence": "Detection Confidence",
        "resolution": "Inference Resolution",
        "tracking": "Object Tracking on karein",
        "summary": "Detected Class Summary dikhayen",
        "tip": "Fast performance ke liye yolov8n.pt use karein. Better accuracy ke liye yolov8s.pt use karein, lekin speed slow ho sakti hai.",
        "execution": "Execution Contexts",
        "image_card": "Image Detection Dashboard",
        "image_desc": "JPG, JPEG ya PNG image upload karein aur objects instantly detect karein.",
        "video_card": "Video Detection Dashboard",
        "video_desc": "MP4, AVI ya MOV video upload karein aur frame by frame detection karein.",
        "live_card": "Live Detection Engine",
        "live_desc": "Webcam se real-time object detection aur tracking karein.",
        "image_tab": "Image Detection",
        "video_tab": "Video Detection",
        "live_tab": "Live Detection",
        "upload_image": "Image upload karein",
        "upload_video": "Video upload karein",
        "run_image": "Image Detection Run karein",
        "run_video": "Video Detection Run karein",
        "original": "Original Image",
        "result": "Detection Result",
        "objects": "Objects Detected",
        "classes": "Unique Classes",
        "conf_level": "Confidence Level",
        "detected_classes": "Detected Classes",
        "image_info": "Object detection start karne ke liye image upload karein.",
        "video_info": "Video detection start karne ke liye video upload karein.",
        "live_info": "Neeche webcam start karein. Best performance ke liye yolov8n.pt aur medium resolution use karein.",
        "processing": "Video process ho rahi hai. Please wait...",
        "processed": "Processed Detection Video",
        "download": "Processed Video Download karein",
        "footer": "AeroVision AI · Object Detection & Tracking Tool · Built with Streamlit, YOLO and OpenCV"
    },
    "UR": {
        "hero_title": "AeroVision AI Object Detection",
        "hero_subtitle": "تصاویر، ویڈیوز اور لائیو کیمرہ کے لیے AI detection اور tracking dashboard۔",
        "overview": "یہ پلیٹ فارم YOLO vision model استعمال کرتا ہے تاکہ images، videos اور webcam سے objects detect اور track کیے جا سکیں۔ آپ confidence اور resolution control بھی set کر سکتے ہیں۔",
        "products": "Products",
        "services": "Services",
        "about": "About",
        "process": "Process",
        "contact": "Contact",
        "menu_services": "Image detection، video detection، object tracking اور live camera detection۔",
        "menu_about": "Streamlit، YOLO اور OpenCV سے بنا ہوا professional AI vision platform۔",
        "menu_process": "Input upload کریں، confidence اور resolution choose کریں، detection run کریں، پھر result download کریں۔",
        "menu_contact": "Developer Contact: dev@aerovision.ai",
        "control": "Control Center",
        "model": "YOLO Model منتخب کریں",
        "confidence": "Detection Confidence",
        "resolution": "Inference Resolution",
        "tracking": "Object Tracking فعال کریں",
        "summary": "Detected Class Summary دکھائیں",
        "tip": "Fast performance کے لیے yolov8n.pt استعمال کریں۔ Better accuracy کے لیے yolov8s.pt استعمال کریں مگر speed slow ہو سکتی ہے۔",
        "execution": "Execution Contexts",
        "image_card": "Image Detection Dashboard",
        "image_desc": "JPG، JPEG یا PNG image upload کریں اور objects detect کریں۔",
        "video_card": "Video Detection Dashboard",
        "video_desc": "MP4، AVI یا MOV video upload کریں اور frame by frame detection کریں۔",
        "live_card": "Live Detection Engine",
        "live_desc": "Webcam سے real-time object detection اور tracking کریں۔",
        "image_tab": "Image Detection",
        "video_tab": "Video Detection",
        "live_tab": "Live Detection",
        "upload_image": "Image upload کریں",
        "upload_video": "Video upload کریں",
        "run_image": "Image Detection چلائیں",
        "run_video": "Video Detection چلائیں",
        "original": "Original Image",
        "result": "Detection Result",
        "objects": "Objects Detected",
        "classes": "Unique Classes",
        "conf_level": "Confidence Level",
        "detected_classes": "Detected Classes",
        "image_info": "Object detection شروع کرنے کے لیے image upload کریں۔",
        "video_info": "Video detection شروع کرنے کے لیے video upload کریں۔",
        "live_info": "نیچے webcam start کریں۔ بہتر performance کے لیے yolov8n.pt اور medium resolution استعمال کریں۔",
        "processing": "Video process ہو رہی ہے۔ Please wait...",
        "processed": "Processed Detection Video",
        "download": "Processed Video Download کریں",
        "footer": "AeroVision AI · Object Detection & Tracking Tool · Built with Streamlit, YOLO and OpenCV"
    }
}


def t(key):
    return TEXT[st.session_state.language_key][key]


# =========================================================
# COLORS
# =========================================================
if st.session_state.theme_mode == "dark":
    BG = "#080B12"
    BG2 = "#0B1018"
    CARD = "#111722"
    CARD2 = "#182232"
    TEXT_COLOR = "#EAF3FF"
    MUTED = "#A9B7C8"
    BORDER = "rgba(189, 221, 252, 0.18)"
    SOFT = "rgba(189, 221, 252, 0.08)"
    SHADOW = "rgba(0,0,0,0.35)"
else:
    # Soft bright mode: not pure white, professional blue-gray look
    BG = "#DDEBF7"
    BG2 = "#CFE2F3"
    CARD = "#EEF7FF"
    CARD2 = "#D9ECFA"
    TEXT_COLOR = "#102235"
    MUTED = "#405D75"
    BORDER = "rgba(56, 73, 89, 0.22)"
    SOFT = "rgba(106, 137, 167, 0.16)"
    SHADOW = "rgba(56,73,89,0.18)"

BLUE_DARK = "#384959"
BLUE_MID = "#6A89A7"
BLUE_LIGHT = "#88BDF2"
BLUE_SOFT = "#BDDDFC"


# =========================================================
# GLOBAL STREAMLIT CSS
# =========================================================
st.markdown(
    f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at top left, rgba(136,189,242,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(106,137,167,0.14), transparent 24%),
        linear-gradient(135deg, {BG} 0%, {BG2} 52%, {BG} 100%);
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1260px;
}}

.stButton > button {{
    border-radius: 14px;
    padding: 12px 14px;
    font-weight: 800;
    color: {TEXT_COLOR};
    background: {CARD};
    border: 1px solid {BORDER};
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, {BLUE_MID}, {BLUE_LIGHT});
    border: 1px solid {BLUE_LIGHT};
    color: #07111C;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 12px;
    margin-top: 14px;
}}

.stTabs [data-baseweb="tab"] {{
    background: {SOFT};
    border-radius: 14px;
    padding: 14px 18px;
    border: 1px solid {BORDER};
    color: {TEXT_COLOR};
    font-weight: 800;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {BLUE_MID}, {BLUE_LIGHT});
    color: #07111C !important;
}}
</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HTML COMPONENT RENDERER
# =========================================================
def html_component(body, height=180):
    st_html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{
    box-sizing: border-box;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    color: {TEXT_COLOR};
}}

.brand-card {{
    display: inline-flex;
    align-items: center;
    gap: 14px;
    padding: 14px 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, {CARD}, {CARD2});
    border: 1px solid {BORDER};
    border-left: 5px solid {BLUE_LIGHT};
    box-shadow: 0 18px 45px {SHADOW};
}}

.logo-mark {{
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: linear-gradient(135deg, {BLUE_SOFT}, {BLUE_LIGHT});
    display: grid;
    place-items: center;
    color: #07111C;
    font-weight: 900;
    font-size: 18px;
    box-shadow: 0 12px 30px rgba(136,189,242,0.30);
}}

.logo-name {{
    font-size: 26px;
    font-weight: 900;
    color: {TEXT_COLOR};
    letter-spacing: -0.8px;
}}

.logo-name span {{
    color: {BLUE_LIGHT};
}}

.logo-sub {{
    color: {MUTED};
    font-size: 13px;
    margin-top: 2px;
    font-weight: 600;
}}

.hero {{
    width: 100%;
    padding: 30px;
    border-radius: 26px;
    background: linear-gradient(135deg, {CARD}, {CARD2});
    border: 1px solid {BORDER};
    box-shadow: 0 22px 55px {SHADOW};
}}

.kicker {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border-radius: 999px;
    background: {SOFT};
    color: {BLUE_SOFT};
    border: 1px solid {BORDER};
    font-size: 13px;
    font-weight: 850;
    margin-bottom: 18px;
}}

.headline-box {{
    padding: 26px;
    border-radius: 22px;
    background: rgba(189,221,252,0.055);
    border: 1px solid {BORDER};
    margin-bottom: 20px;
}}

.hero-title {{
    color: {TEXT_COLOR};
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.05;
    font-weight: 950;
    letter-spacing: -1.8px;
    margin-bottom: 14px;
}}

.hero-title span {{
    color: {BLUE_LIGHT};
}}

.hero-subtitle {{
    color: {MUTED};
    font-size: 17px;
    line-height: 1.7;
    font-weight: 650;
}}

.hero-overview {{
    color: {MUTED};
    font-size: 16px;
    line-height: 1.8;
}}

.feature-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 22px;
}}

.feature-pill {{
    padding: 10px 14px;
    border-radius: 999px;
    color: {TEXT_COLOR};
    background: {SOFT};
    border: 1px solid {BORDER};
    font-size: 13px;
    font-weight: 850;
}}

.menu-card {{
    width: 100%;
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, {CARD}, {CARD2});
    border: 1px solid {BORDER};
    box-shadow: 0 22px 55px {SHADOW};
}}

.menu-title {{
    color: {TEXT_COLOR};
    font-size: 20px;
    font-weight: 900;
    margin-bottom: 16px;
}}

.menu-link {{
    color: {TEXT_COLOR};
    font-size: 16px;
    font-weight: 850;
    padding-top: 12px;
}}

.menu-small {{
    color: {MUTED};
    font-size: 13px;
    line-height: 1.7;
    margin-top: 5px;
    margin-bottom: 12px;
}}

.menu-divider {{
    height: 1px;
    background: {BORDER};
    margin: 16px 0;
}}

.section-title {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 27px;
    font-weight: 900;
    color: {TEXT_COLOR};
    margin-bottom: 18px;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}}

.card {{
    padding: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, {CARD}, {CARD2});
    border: 1px solid {BORDER};
    min-height: 170px;
    box-shadow: 0 15px 35px {SHADOW};
}}

.card-icon {{
    font-size: 34px;
    margin-bottom: 18px;
}}

.card-title {{
    color: {TEXT_COLOR};
    font-weight: 900;
    font-size: 19px;
    margin-bottom: 10px;
}}

.card-desc {{
    color: {MUTED};
    font-size: 14px;
    line-height: 1.7;
}}

.splash-screen {{
    min-height: 80vh;
    display: grid;
    place-items: center;
    text-align: center;
}}

.splash-logo {{
    width: 115px;
    height: 115px;
    border-radius: 32px;
    background: linear-gradient(135deg, {BLUE_SOFT}, {BLUE_LIGHT});
    display: grid;
    place-items: center;
    color: #07111C;
    font-size: 40px;
    font-weight: 950;
    margin: 0 auto 28px auto;
    box-shadow: 0 24px 70px rgba(136,189,242,0.35);
}}

.splash-title {{
    color: {TEXT_COLOR};
    font-size: clamp(28px, 5vw, 50px);
    font-weight: 950;
    letter-spacing: 8px;
    margin-bottom: 14px;
}}

.splash-sub {{
    color: {MUTED};
    font-size: 16px;
    font-weight: 750;
    letter-spacing: 2px;
}}

.footer-box {{
    padding: 16px;
    border-radius: 16px;
    background: {SOFT};
    border: 1px solid {BORDER};
    color: {MUTED};
    text-align: center;
    font-size: 13px;
}}

@media(max-width: 768px) {{
    .cards {{
        grid-template-columns: 1fr;
    }}

    .logo-name {{
        font-size: 21px;
    }}

    .logo-sub {{
        font-size: 11px;
    }}

    .hero {{
        padding: 22px;
    }}

    .headline-box {{
        padding: 20px;
    }}
}}
</style>
</head>
<body>
{body}
</body>
</html>
""",
        height=height,
        scrolling=False
    )


# =========================================================
# SPLASH SCREEN
# =========================================================
if not st.session_state.splash_done:
    html_component(
        """
<div class="splash-screen">
    <div>
        <div class="splash-logo">AV</div>
        <div class="splash-title">AEROVISION AI</div>
        <div class="splash-sub">INITIALIZING VISION SYSTEMS...</div>
    </div>
</div>
""",
        height=520
    )

    progress = st.progress(0)
    loading_text = st.empty()

    for percent in range(1, 101):
        progress.progress(percent)
        loading_text.caption(f"Loading detection engine... {percent}%")
        time.sleep(0.012)

    st.session_state.splash_done = True
    st.rerun()


# =========================================================
# YOLO FUNCTIONS
# =========================================================
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(model_name)


def run_detection(model, frame, conf_value, resolution, use_tracking=False, target_class_ids=None):
    try:
        if use_tracking:
            return model.track(
                frame,
                conf=conf_value,
                imgsz=resolution,
                classes=target_class_ids,
                persist=True,
                verbose=False
            )

        return model.predict(
            frame,
            conf=conf_value,
            imgsz=resolution,
            classes=target_class_ids,
            verbose=False
        )

    except Exception:
        return model.predict(
            frame,
            conf=conf_value,
            imgsz=resolution,
            classes=target_class_ids,
            verbose=False
        )



def detect_image(image, model, conf_value, resolution, target_class_ids=None):
    image_array = np.array(image.convert("RGB"))
    results = run_detection(
        model,
        image_array,
        conf_value,
        resolution,
        False,
        target_class_ids
    )
    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted, results



def count_detections(results):
    if not results:
        return 0, []

    boxes = results[0].boxes

    if boxes is None:
        return 0, []

    names = results[0].names
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    detected_names = [names[int(cls)] for cls in classes]

    return len(detected_names), detected_names


def get_target_class_ids(model, target_text):
    if not target_text or target_text.strip() == "":
        return None, []

    user_items = [
        item.strip().lower()
        for item in target_text.split(",")
        if item.strip()
    ]

    aliases = {
        "phone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "smartphone": "cell phone",
        "human": "person",
        "man": "person",
        "woman": "person",
        "people": "person",
        "bike": "bicycle",
        "cycle": "bicycle",
        "motorbike": "motorcycle",
        "motor cycle": "motorcycle",
        "tv": "tv",
        "television": "tv"
    }

    matched_ids = []
    matched_names = []

    for user_item in user_items:
        search_item = aliases.get(user_item, user_item)

        for class_id, class_name in model.names.items():
            class_name_lower = class_name.lower()

            if search_item == class_name_lower or search_item in class_name_lower:
                if class_id not in matched_ids:
                    matched_ids.append(class_id)
                    matched_names.append(class_name)

    return matched_ids if matched_ids else None, matched_names



def process_video(video_path, output_path, model, conf_value, resolution, use_tracking, target_class_ids=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps is None:
        fps = 24

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = run_detection(
            model,
            frame,
            conf_value,
            resolution,
            use_tracking,
            target_class_ids
        )
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        current_frame += 1

        if frame_count > 0:
            progress_value = min(current_frame / frame_count, 1.0)
            progress_bar.progress(progress_value)
            status_text.write(f"Processing video frame {current_frame}/{frame_count}")

    cap.release()
    writer.release()

    progress_bar.progress(1.0)
    status_text.success("Video processing completed successfully.")

    return True



class VideoProcessor:
    def __init__(self, model_name):
        self.model = load_yolo_model(model_name)
        self.confidence = 0.35
        self.resolution = 640
        self.use_tracking = True
        self.target_class_ids = None
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            conf = self.confidence
            resolution = self.resolution
            tracking = self.use_tracking
            target_class_ids = self.target_class_ids

        results = run_detection(
            self.model,
            image,
            conf,
            resolution,
            tracking,
            target_class_ids
        )
        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")



# =========================================================
# TOP BRAND + MENU BUTTON
# =========================================================
nav_left, nav_right = st.columns([7.8, 0.7])

with nav_left:
    html_component(
        """
<div class="brand-card">
    <div class="logo-mark">AV</div>
    <div>
        <div class="logo-name">AeroVision <span>AI</span></div>
        <div class="logo-sub">Object Detection & Tracking Platform</div>
    </div>
</div>
""",
        height=110
    )

with nav_right:
    st.write("")
    if st.button("⋮", help="Menu"):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()


# =========================================================
# COMPACT MENU
# =========================================================
if st.session_state.menu_open:
    menu_left, menu_right = st.columns([5.8, 2.2])

    with menu_right:
        html_component(
            """
<div class="menu-card">
    <div class="menu-title">Menu</div>
    <div class="menu-small">Choose an option below.</div>
</div>
""",
            height=95
        )

        if st.button("🧩 " + t("products"), use_container_width=True):
            st.session_state.menu_loading_item = "products"
            st.session_state.active_menu_item = None
            st.rerun()

        if st.button("⚙️ " + t("services"), use_container_width=True):
            st.session_state.menu_loading_item = "services"
            st.session_state.active_menu_item = None
            st.rerun()

        if st.button("ℹ️ " + t("about"), use_container_width=True):
            st.session_state.menu_loading_item = "about"
            st.session_state.active_menu_item = None
            st.rerun()

        if st.button("🔄 " + t("process"), use_container_width=True):
            st.session_state.menu_loading_item = "process"
            st.session_state.active_menu_item = None
            st.rerun()

        if st.button("📩 " + t("contact"), use_container_width=True):
            st.session_state.menu_loading_item = "contact"
            st.session_state.active_menu_item = None
            st.rerun()

        if st.session_state.menu_loading_item:
            loading_item = st.session_state.menu_loading_item
            loading_label = t(loading_item) if loading_item in TEXT[st.session_state.language_key] else loading_item.title()

            html_component(
                f"""
<div class="menu-loader">
    <div class="menu-loader-logo">AV</div>
    <div class="menu-loader-title">AEROVISION AI</div>
    <div class="menu-loader-sub">OPENING {escape(loading_label).upper()}...</div>
</div>
""",
                height=215
            )

            menu_progress = st.progress(0)
            for percent in range(1, 101):
                menu_progress.progress(percent)
                time.sleep(0.006)

            st.session_state.active_menu_item = loading_item
            st.session_state.menu_loading_item = None
            st.rerun()

        detail_title = None
        detail_text = None

        if st.session_state.active_menu_item == "products":
            detail_title = "🧩 " + t("products")
            detail_text = "AeroVision AI supports image detection, video detection, object tracking, and live camera AI vision."
        elif st.session_state.active_menu_item == "services":
            detail_title = "⚙️ " + t("services")
            detail_text = t("menu_services")
        elif st.session_state.active_menu_item == "about":
            detail_title = "ℹ️ " + t("about")
            detail_text = t("menu_about")
        elif st.session_state.active_menu_item == "process":
            detail_title = "🔄 " + t("process")
            detail_text = t("menu_process")
        elif st.session_state.active_menu_item == "contact":
            detail_title = "📩 " + t("contact")
            detail_text = t("menu_contact")

        if detail_title and detail_text:
            html_component(
                f"""
<div class="menu-card">
    <div class="menu-link">{escape(detail_title)}</div>
    <div class="menu-small">{escape(detail_text)}</div>
</div>
""",
                height=140
            )

        selected_language = st.selectbox(
            "🌐 Language",
            options=list(LANGUAGE_NAMES.keys()),
            index=list(LANGUAGE_NAMES.keys()).index(st.session_state.language_key),
            format_func=lambda key: LANGUAGE_NAMES[key],
        )

        if selected_language != st.session_state.language_key:
            st.session_state.language_key = selected_language
            st.rerun()

        theme_label = "☀️ Light Mode" if st.session_state.theme_mode == "dark" else "🌙 Dark Mode"

        if st.button(theme_label, use_container_width=True):
            st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
            st.rerun()


# =========================================================
# HERO SECTION
# =========================================================
hero_title = escape(t("hero_title")).replace("Object Detection", "<span>Object Detection</span>")

html_component(
    f"""
<div class="hero">
    <div class="kicker">🧠 YOLO Vision Model • Image • Video • Live Camera</div>

    <div class="headline-box">
        <div class="hero-title">{hero_title}</div>
        <div class="hero-subtitle">{escape(t("hero_subtitle"))}</div>
    </div>

    <div class="hero-overview">{escape(t("overview"))}</div>

    <div class="feature-row">
        <div class="feature-pill">🧠 YOLO Vision Model</div>
        <div class="feature-pill">🖼️ Image Detection</div>
        <div class="feature-pill">🎬 Video Tracking</div>
        <div class="feature-pill">📹 Live Camera</div>
    </div>
</div>
""",
    height=430
)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.markdown(f"## ⚙️ {t('control')}")

    model_choice = st.selectbox(
        t("model"),
        ["yolov8n.pt", "yolov8s.pt"],
        index=0
    )

    confidence = st.slider(
        t("confidence"),
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    resolution = st.slider(
        t("resolution"),
        min_value=320,
        max_value=1280,
        value=640,
        step=160
    )

    enable_tracking = st.checkbox(t("tracking"), value=True)
    show_classes = st.checkbox(t("summary"), value=True)

    st.markdown("---")
    st.info(t("tip"))


with st.spinner("Loading YOLO model..."):
    model = load_yolo_model(model_choice)


# =========================================================
# EXECUTION CARDS
# =========================================================
html_component(
    f"""
<div class="section-title">⚡ {escape(t("execution"))}</div>

<div class="cards">
    <div class="card">
        <div class="card-icon">🖼️</div>
        <div class="card-title">{escape(t("image_card"))}</div>
        <div class="card-desc">{escape(t("image_desc"))}</div>
    </div>

    <div class="card">
        <div class="card-icon">🎬</div>
        <div class="card-title">{escape(t("video_card"))}</div>
        <div class="card-desc">{escape(t("video_desc"))}</div>
    </div>

    <div class="card">
        <div class="card-icon">📹</div>
        <div class="card-title">{escape(t("live_card"))}</div>
        <div class="card-desc">{escape(t("live_desc"))}</div>
    </div>
</div>
""",
    height=280
)


# =========================================================
# MAIN TABS
# =========================================================
tab_image, tab_video, tab_live = st.tabs(
    [
        f"🖼️ {t('image_tab')}",
        f"🎬 {t('video_tab')}",
        f"📹 {t('live_tab')}"
    ]
)


# =========================================================
# IMAGE DETECTION
# =========================================================
with tab_image:
    st.markdown(f"### 🖼️ {t('image_tab')}")

    uploaded_image = st.file_uploader(
        t("upload_image"),
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    image_target = st.text_input(
        "🎯 What do you want to detect in this image?",
        placeholder="Example: person, car, cell phone",
        key="image_target_input"
    )

    image_target_ids, image_matched_names = get_target_class_ids(model, image_target)

    if image_target.strip() and not image_matched_names:
        st.warning("No matching YOLO class found. Try: person, car, cell phone, dog, bottle, laptop.")
    elif image_matched_names:
        st.success("Detecting only: " + ", ".join(image_matched_names))

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        col_original, col_result = st.columns(2)

        with col_original:
            st.markdown(f"#### {t('original')}")
            st.image(image, use_container_width=True)

        if st.button(f"🚀 {t('run_image')}"):
            with st.spinner("Detecting objects in image..."):
                result_image, results = detect_image(
                    image,
                    model,
                    confidence,
                    resolution,
                    image_target_ids
                )

                total_objects, detected_names = count_detections(results)

            with col_result:
                st.markdown(f"#### {t('result')}")
                st.image(result_image, use_container_width=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(t("objects"), total_objects)

            with m2:
                st.metric(t("conf_level"), f"{confidence:.2f}")

            with m3:
                st.metric(t("classes"), len(set(detected_names)))

            if show_classes and detected_names:
                st.markdown(f"#### {t('detected_classes')}")
                st.write(", ".join(sorted(set(detected_names))))
    else:
        st.info(t("image_info"))


# =========================================================
# VIDEO DETECTION
# =========================================================
with tab_video:
    st.markdown(f"### 🎬 {t('video_tab')}")

    uploaded_video = st.file_uploader(
        t("upload_video"),
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    video_target = st.text_input(
        "🎯 What do you want to detect in this video?",
        placeholder="Example: person, car, dog",
        key="video_target_input"
    )

    video_target_ids, video_matched_names = get_target_class_ids(model, video_target)

    if video_target.strip() and not video_matched_names:
        st.warning("No matching YOLO class found. Try: person, car, cell phone, dog, bottle, laptop.")
    elif video_matched_names:
        st.success("Detecting only: " + ", ".join(video_matched_names))

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button(f"🎞️ {t('run_video')}"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                temp_input.write(uploaded_video.read())
                input_video_path = temp_input.name

            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            with st.spinner(t("processing")):
                success = process_video(
                    input_video_path,
                    output_video_path,
                    model,
                    confidence,
                    resolution,
                    enable_tracking,
                    video_target_ids
                )

            if success:
                st.markdown(f"#### {t('processed')}")
                st.video(output_video_path)

                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ {t('download')}",
                        data=file,
                        file_name="aerovision_detected_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Video could not be processed. Try another video file.")

            try:
                os.remove(input_video_path)
            except Exception:
                pass
    else:
        st.info(t("video_info"))


# =========================================================
# LIVE DETECTION
# =========================================================
with tab_live:
    st.markdown(f"### 📹 {t('live_tab')}")
    st.info(t("live_info"))

    live_target = st.text_input(
        "🎯 What do you want to detect in live camera?",
        placeholder="Example: person, car, phone",
        key="live_target_input"
    )

    live_target_ids, live_matched_names = get_target_class_ids(model, live_target)

    if live_target.strip() and not live_matched_names:
        st.warning("No matching YOLO class found. Try: person, car, cell phone, dog, bottle, laptop.")
    elif live_matched_names:
        st.success("Detecting only: " + ", ".join(live_matched_names))

    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc is not installed. Run: pip install streamlit-webrtc av")
    else:
        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            }
        )

        ctx = webrtc_streamer(
            key=f"aerovision-live-detection-{model_choice}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: VideoProcessor(model_choice),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )

        if ctx.video_processor:
            with ctx.video_processor.lock:
                ctx.video_processor.confidence = confidence
                ctx.video_processor.resolution = resolution
                ctx.video_processor.use_tracking = enable_tracking
                ctx.video_processor.target_class_ids = live_target_ids


# =========================================================
# FOOTER
# =========================================================
html_component(
    f"""
<div class="footer-box">
    {escape(t("footer"))}
</div>
""",
    height=70
)