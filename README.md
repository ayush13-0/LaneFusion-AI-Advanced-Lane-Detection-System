<!-- PROJECT BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/LaneFusion%20AI-Advanced%20Lane%20Detection%20System-blueviolet?style=for-the-badge&logo=python&logoColor=white" />
</p>

<!-- TITLE -->
<h1 align="center">🚗🤖 LaneFusion AI – Advanced Lane Detection System</h1>

<!-- TAGLINE -->
<p align="center">
  <b>An advanced Computer Vision & Deep Learning–based lane detection system using OpenCV, U-Net & Streamlit</b>
</p>

<!-- CORE TECH BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-U--Net-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit" />
</p>

<!-- ADVANCED / SYSTEM BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Model-Classical%20CV%20%2B%20U--Net-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Application-ADAS%20%7C%20Autonomous%20Driving-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" />
</p>

<!-- DEPLOYMENT / INTEGRATION BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20App-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Integration-ROS%20%7C%20CARLA%20%7C%20LGSVL-blue?style=for-the-badge" />
</p>

<!-- OPTIONAL DEMO BADGE -->
<p align="center">
  <img src="https://img.shields.io/badge/Live%20Demo-Available-success?style=for-the-badge&logo=streamlit" />
</p>

# 📌 Overview

**LaneFusion Pro** is a fully-featured lane detection system built using Computer Vision, Image Processing, and Deep Learning (U-Net).
- The project delivers a modern and interactive Streamlit dashboard that can:
- Detect lanes on images, videos, dashcam footage, and webcam streams
- Run on traditional CV (Canny + Hough, Perspective Transform, Polynomial Fitting)
- Run on deep learning (U-Net segmentation)
- Visualize lane curvature, vehicle position offset, masks, and intermediary pipeline outputs
- Export processed video results
This makes LaneFusion Pro suitable for Autonomous Driving, Driver Assist Systems, Robotics, Simulations, and Research.

# 📁 Project Structure
<pre> 
  LaneFusion/
  │
  ├── masks_png/                 # PNG lane masks
  ├── road_line_annotation/      # (Looks unused or raw annotations)
  ├── road_line_images/          # Source images for training/testing
  │
  ├── LaneFusion2.ipynb          # Notebook 1
  ├── LaneFusionAI-Main.ipynb    # Notebook 2 (Main research/training pipeline)
  │
  ├── LaneFusionCV_UNET.h5       # Your trained U-Net model
  ├── LF-AI.py                   # Python script (maybe inference?)
  │
  ├── requirements.txt
  └── README.md   (to be created)  </pre>




# 🚀 Features
🟦 **1. Computer Vision Lane Detection** 
- Gaussian blur → Canny edge detection
- Region of Interest masking
- Hough Line Transform
- Perspective (bird’s-eye) transform
- Sliding-window lane search
- 2nd-degree polynomial lane fitting
- Curvature radius + vehicle offset estimation
- Smoothing across frames

🟩 **2. Deep Learning Lane Segmentation (U-Net)**
Load trained U-Net .h5 model
Generate segmentation masks
Overlay masks on original frame
Adjustable confidence thresholds
Works on image & video inputs

🟦 **3. Streamlit App (Pro Version)**
*Modern sidebar controls*
**Light/Dark theme**
Tabs for:
- Image processing
- Video pipeline
- Webcam stream
- Diagnostics
- Deep Learning mode
- Video export with progress bar
- Batch processing for entire folders
- Smooth animations and error handling

🔧 **4. Deployment & Integration**
- ROS-friendly architecture
- Docker-ready folder structure
- Easy to connect with car simulators (CARLA, LGSVL)

# 🛠 Installation
1. **Clone the repository**
<pre> git clone https://github.com/ayush13-0/LaneFusion-AI-Pro-Advanced-Lane-Detection-System
cd LaneFusion-AI-Pro-Advanced-Lane-Detection-System </pre>

2. **Install dependencies**
<pre> pip install -r requirements.txt </pre>

▶️ Run the Streamlit App
streamlit run LF-AI.py

#🎥 Demo
**(Add screenshots or GIFs here once available)**

# 🧠 Deep Learning Model (U-Net)
**LaneFusion Pro supports loading a U-Net model:**
<pre> from unet_model import load_unet_model
model = load_unet_model("models/lane_unet.h5") </pre>
- To train your own model, provide:
- Image folder
- Mask folder
- Augmentations (optional)

# 🔍 Tech Stack
- Area	Tools
- CV Pipeline	OpenCV, NumPy
- Deep Learning	TensorFlow / Keras
- UI	Streamlit
- Video Handling	imageio, cv2.VideoCapture
- Visualization	Matplotlib, OpenCV overlays

<!-- 🧪 LaneFusion AI – Architecture Diagram -->
<h2 align="center">🧪 LaneFusion AI – System Architecture</h2>

<p align="center">
<svg width="900" height="520" viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg">

  <!-- Background -->
  <rect x="0" y="0" width="900" height="520" rx="16" fill="#0d1117"/>

  <!-- Title -->
  <text x="450" y="40" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="bold">
    Lane Detection Architecture (CV + U-Net)
  </text>

  <!-- Input -->
  <rect x="360" y="70" width="180" height="40" rx="8" fill="#238636"/>
  <text x="450" y="95" text-anchor="middle" fill="#ffffff" font-size="14">Input Frame</text>

  <!-- CV Pipeline -->
  <rect x="80" y="150" width="260" height="300" rx="12" fill="#161b22" stroke="#1f6feb" stroke-width="2"/>
  <text x="210" y="175" text-anchor="middle" fill="#1f6feb" font-size="16" font-weight="bold">
    Classical CV Pipeline
  </text>

  <text x="100" y="205" fill="#c9d1d9" font-size="13">
    • Undistortion
  </text>
  <text x="100" y="230" fill="#c9d1d9" font-size="13">
    • ROI Masking
  </text>
  <text x="100" y="255" fill="#c9d1d9" font-size="13">
    • Canny Edge Detection
  </text>
  <text x="100" y="280" fill="#c9d1d9" font-size="13">
    • Hough Transform
  </text>
  <text x="100" y="305" fill="#c9d1d9" font-size="13">
    • Perspective Transform
  </text>
  <text x="100" y="330" fill="#c9d1d9" font-size="13">
    • Sliding Window Search
  </text>
  <text x="100" y="355" fill="#c9d1d9" font-size="13">
    • Polynomial Curve Fit
  </text>
  <text x="100" y="380" fill="#c9d1d9" font-size="13">
    • Lane Visualization
  </text>

  <!-- U-Net Pipeline -->
  <rect x="560" y="150" width="260" height="300" rx="12" fill="#161b22" stroke="#2ea043" stroke-width="2"/>
  <text x="690" y="175" text-anchor="middle" fill="#2ea043" font-size="16" font-weight="bold">
    Deep Learning Pipeline (U-Net)
  </text>

  <text x="580" y="215" fill="#c9d1d9" font-size="13">
    • Preprocessing
  </text>
  <text x="580" y="245" fill="#c9d1d9" font-size="13">
    • U-Net Segmentation
  </text>
  <text x="580" y="275" fill="#c9d1d9" font-size="13">
    • Binary Lane Mask
  </text>
  <text x="580" y="305" fill="#c9d1d9" font-size="13">
    • Mask Overlay
  </text>
  <text x="580" y="335" fill="#c9d1d9" font-size="13">
    • Final Lane Output
  </text>

  <!-- Arrows -->
  <line x1="450" y1="110" x2="210" y2="150" stroke="#8b949e" stroke-width="2"/>
  <line x1="450" y1="110" x2="690" y2="150" stroke="#8b949e" stroke-width="2"/>

</svg>
</p>


# 📦 Future Enhancements
- Lane departure warnings
- Vehicle detection + lane fusion
- YOLOv9 integration
- Temporal smoothing with Kalman Filters / Optical Flow
- ONNX Runtime / TensorRT acceleration
- Mobile app version with Streamlit Cloud

# 🤝 Contributing
Contributions are welcome!
Please open a PR or create an issue with a feature suggestion.

⭐ Support
If you like this project, consider giving it a star ⭐ on GitHub — it helps a lot!
