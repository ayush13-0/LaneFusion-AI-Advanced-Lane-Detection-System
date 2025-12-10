# <p align="center">🚗🤖 LaneFusion AI – Advanced Lane Detection System (CV + U-Net + Streamlit App)</p>
<p align="center"> <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Deep%20Learning-U--Net-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/App-Streamlit-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge"/> </p>

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

# 🧪 How It Works
1. CV Pipeline
flowchart LR
A[Frame] --> B[Undistort]
B --> C[ROI Mask]
C --> D[Canny Edge Detection]
D --> E[Hough Transform]
E --> F[Perspective Transform]
F --> G[Sliding Window Search]
G --> H[Polynomial Fit]
H --> I[Lane Visualization]

2. U-Net Pipeline
flowchart LR
A[Frame] --> B[Preprocess]
B --> C[U-Net Segmentation]
C --> D[Generate Mask]
D --> E[Overlay on Frame]

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
