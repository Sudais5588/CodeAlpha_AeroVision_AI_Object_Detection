# AeroVision AI – Object Detection & Tracking Tool

AeroVision AI is a professional **AI-powered object detection and tracking dashboard** built with **Python, Streamlit, YOLO, OpenCV, and WebRTC**.  
The tool allows users to detect objects from **images**, **uploaded videos**, and **live camera streams** through a modern, responsive, and portfolio-ready interface.

This project is designed as an internship / portfolio submission project and demonstrates practical computer vision skills, real-time detection, user-controlled detection settings, and a clean UI/UX.

---
## 🔗 Live Demo

https://aerovision-ai-object-detection.streamlit.app/




## 🚀 Live Project Overview

AeroVision AI provides three main detection modes:

1. **Image Detection**
   - Upload an image.
   - Enter a target object if needed.
   - Run YOLO detection.
   - View detected objects with bounding boxes and confidence scores.

2. **Video Detection**
   - Upload a video file.
   - Detect objects frame by frame.
   - Enable or disable object tracking.
   - Download the processed video output.

3. **Live Camera Detection**
   - Use webcam for real-time object detection.
   - Detect all objects or only a specific user-defined object.
   - Supports confidence and resolution control.

---

## ✨ Key Features

### 🖼️ Image Object Detection
- Supports `JPG`, `JPEG`, and `PNG` images.
- Displays original image and detection result side by side.
- Shows detected object count.
- Shows confidence level.
- Shows unique detected classes.
- Allows user to detect a specific object only.

Example target input:

```text
person
car
cell phone
dog
bottle
laptop
```

---

### 🎬 Video Object Detection
- Supports video formats such as `MP4`, `AVI`, and `MOV`.
- Processes videos frame by frame.
- Draws bounding boxes on detected objects.
- Allows processed video download.
- Supports object tracking mode.
- Allows targeted detection in videos.

---

### 📹 Live Camera Detection
- Real-time object detection through webcam.
- Built using `streamlit-webrtc`.
- Supports live object tracking.
- Allows targeted live detection.
- Useful for real-time computer vision demonstrations.

---

### 🎯 Target Object Detection
AeroVision AI does not only detect all objects.  
It also allows the user to enter a specific object name and detect only that object.

For example:

```text
What do you want to detect?
car
```

Then the system detects only cars and ignores other objects.

Supported examples include:

```text
person
human
car
bus
truck
dog
cat
cell phone
phone
mobile
laptop
bottle
chair
motorcycle
bicycle
```

---

### ⚙️ Control Center
The sidebar control center allows the user to configure detection settings.

Available controls:

- YOLO model selection
- Detection confidence
- Inference resolution
- Object tracking toggle
- Detected class summary toggle

---

### 🌗 Dark and Light Mode
The app includes two professional themes:

- Dark mode
- Soft bright mode

The light mode is designed with a soft blue-white color palette to avoid harsh brightness.

---

### 🌐 Multi-language Support
The interface supports:

- English
- Roman Urdu
- Urdu

This improves usability for local and international users.

---

### 📌 Modern UI / UX
The design includes:

- Professional AeroVision AI branding
- Splash loading screen
- Clean menu system
- Smooth dark/light mode switching
- Modern cards and dashboard layout
- Responsive design
- Portfolio-ready color palette

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web dashboard framework |
| Ultralytics YOLO | Object detection model |
| OpenCV | Image and video processing |
| Pillow | Image handling |
| NumPy | Array and image operations |
| streamlit-webrtc | Live camera streaming |
| AV | Video frame processing |
| HTML/CSS | Custom UI styling |

---

## 🧠 Model Used

This project uses **YOLOv8** from the Ultralytics library.

Available model options in the app:

```text
yolov8n.pt
yolov8s.pt
```

### Recommended Model

For normal laptops and faster performance:

```text
yolov8n.pt
```

For better accuracy but slower performance:

```text
yolov8s.pt
```

---

## 📂 Project Structure

```bash
AeroVision-AI-Object-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── screenshots/
│   ├── home.png
│   ├── image-detection.png
│   ├── video-detection.png
│   └── live-detection.png
│
└── demo/
    └── aerovision-demo.mp4
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AeroVision-AI-Object-Detection.git
```

```bash
cd AeroVision-AI-Object-Detection
```

---

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
```

For macOS/Linux:

```bash
python3 -m venv .venv
```

---

### 3. Activate Virtual Environment

For Windows PowerShell:

```bash
.\.venv\Scripts\activate
```

For Windows CMD:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` file and add:

```txt
streamlit
ultralytics
opencv-python
pillow
numpy
streamlit-webrtc
av
```

---

## 🧪 How to Use the App

### Image Detection

1. Open the app.
2. Go to the **Image Detection** tab.
3. Upload an image.
4. Optional: type the object you want to detect.
5. Click **Run Image Detection**.
6. View the original image and detection result.

---

### Video Detection

1. Go to the **Video Detection** tab.
2. Upload a video file.
3. Optional: type the object you want to detect.
4. Click **Run Video Detection**.
5. Wait for processing.
6. Download the processed output video.

---

### Live Camera Detection

1. Go to the **Live Detection** tab.
2. Optional: type the object you want to detect.
3. Start webcam detection.
4. Allow camera permission in the browser.
5. View real-time detection results.

---

## 🎯 Target Detection Examples

| User Input | YOLO Class Detected |
|---|---|
| phone | cell phone |
| mobile | cell phone |
| human | person |
| man | person |
| woman | person |
| bike | bicycle |
| motorbike | motorcycle |
| car | car |
| dog | dog |
| laptop | laptop |

---

## 📸 Screenshots

Add your screenshots in the `screenshots` folder and update the paths below.

### Home Page

```markdown
![Home Page](screenshots/home.png)
```

### Image Detection

```markdown
![Image Detection](screenshots/image-detection.png)
```

### Video Detection

```markdown
![Video Detection](screenshots/video-detection.png)
```

### Live Detection

```markdown
![Live Detection](screenshots/live-detection.png)
```

---

## 🎥 Demo Video

Add your demo video link here.

```markdown
[Watch Demo Video](YOUR_DEMO_VIDEO_LINK)
```

Example:

```markdown
[Watch Demo Video](demo/aerovision-demo.mp4)
```

---

## 🧩 Main Functional Flow

```text
User Input
   ↓
Image / Video / Live Camera
   ↓
YOLO Model Processing
   ↓
Object Detection / Tracking
   ↓
Bounding Box Visualization
   ↓
Result Display / Download
```

---

## 📊 Project Highlights

- Complete computer vision dashboard
- Real-time webcam support
- User-defined object filtering
- Modern professional UI
- Multi-language interface
- Dark and light mode
- Downloadable video output
- Suitable for internship and portfolio submission

---

## 🚀 Deployment

You can deploy this project on:

- Streamlit Community Cloud
- Render
- Hugging Face Spaces
- Local machine

### Streamlit Cloud Deployment Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path:

```text
app.py
```

6. Click **Deploy**.

---

## 🧑‍💻 Author

**Muhammad Sudais / Hamdan Nasir**  
BS Computer Science – AI Specialization  
AI Engineer / Data Science Enthusiast

---

## 🎓 Internship Submission

This project was developed as part of an internship task submission.  
It demonstrates practical use of AI, computer vision, object detection, object tracking, and deployment-ready web application development.

---

## 📌 Future Improvements

Possible future upgrades:

- Add object counting by category
- Add detection history
- Add CSV report export
- Add object speed estimation
- Add email alert system
- Add database storage
- Add user authentication
- Add multiple YOLO model support
- Add deployment on cloud GPU

---

## ❓ Common Issues

### 1. Webcam is not opening

Try:

- Use Chrome browser
- Allow camera permission
- Run on localhost
- Use HTTPS if deployed online

---

### 2. Video processing is slow

Try:

- Use `yolov8n.pt`
- Reduce inference resolution
- Use shorter video clips
- Close other heavy apps

---

### 3. Module not found error

Run:

```bash
pip install -r requirements.txt
```

---

### 4. YOLO model downloading slowly

The model downloads automatically the first time.  
Wait until the model download completes.

---

## 📄 License

This project is open-source and available for educational and portfolio purposes.

---

## ⭐ Acknowledgement

This project uses:

- Ultralytics YOLO
- Streamlit
- OpenCV
- streamlit-webrtc

Special thanks to the open-source community for providing tools that make AI and computer vision development accessible.
