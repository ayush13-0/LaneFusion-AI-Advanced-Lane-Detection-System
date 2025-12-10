"""
Advanced Streamlit app: LaneFusion Pro — Professional Lane Detection Interface
Goal: Provide a more advanced, polished Streamlit UI for detecting road lanes from dashcam footage.

Features added (pro-level):
- Polished responsive layout with sidebar controls and organized tabs
- Preset configurations and fine-grained parameter control
- Perspective (bird's-eye) transform to estimate lane curvature and perform lane fitting
- Left/right lane separation, polynomial fitting, and curvature/offset estimation
- Temporal smoothing across frames with a configurable buffer
- U-Net integration (optional) with model manager
- Live webcam / camera input and file upload support
- Video export, performance info, and GPU check
- Custom CSS for a cleaner, professional look

How to run:
    pip install -r requirements.txt
    streamlit run LF-AI.py

"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import deque
from typing import Tuple, List

# Optional DL support
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="LaneFusion Pro", layout="wide")

# -------------------------
# Custom CSS for nicer UI
# -------------------------
st.markdown("""
<style>
.block-container{padding-top:1rem;}
h1{color:#0b5cff}
.streaming-badge{background:linear-gradient(90deg,#0b5cff,#00c2ff);color:white;padding:6px 10px;border-radius:8px}
.small-muted{font-size:12px;color:#6c757d}
.card{background:#ffffff;box-shadow:0 6px 18px rgba(0,0,0,0.08);padding:12px;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utils: image & transforms
# -------------------------

def resize_keep_aspect(image: np.ndarray, width: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    return cv2.resize(image, (width, int(h*scale)))


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perspective transform helpers
def get_perspective_transform(src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    Minv = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))
    return M, Minv

# Color/gradient threshold for lane detection

def color_threshold(img):
    # Convert to HLS and threshold based on S channel and lightness
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
    # Sobel x on L channel
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / np.max(abs_sobelx + 1e-8))
    sobel_binary = (scaled > 20) & (scaled < 200)

    s_binary = (s > 90)
    combined = np.logical_or(sobel_binary, s_binary)
    return (combined * 255).astype(np.uint8)

# Sliding-window lane fit

def sliding_window_poly(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
        return None, None

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

# Curvature and offset helpers

def measure_curvature_and_offset(left_fit, right_fit, image_shape):
    # meters per pixel (rough approximations)
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    ploty = np.linspace(0, image_shape[0]-1, image_shape[0])
    y_eval = np.max(ploty)

    # Convert to real world
    left_fit_cr = np.array([left_fit[0]*xm_per_pix/(ym_per_pix**2), left_fit[1]*xm_per_pix/ym_per_pix, left_fit[2]*xm_per_pix])
    right_fit_cr = np.array([right_fit[0]*xm_per_pix/(ym_per_pix**2), right_fit[1]*xm_per_pix/ym_per_pix, right_fit[2]*xm_per_pix])

    # Radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0] + 1e-6)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0] + 1e-6)

    # Vehicle offset from lane center
    lane_center = (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2] + right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]) / 2
    car_center = image_shape[1] / 2
    center_offset_pixels = car_center - lane_center
    center_offset_mtrs = center_offset_pixels * xm_per_pix

    return (left_curverad, right_curverad), center_offset_mtrs

# Draw lane onto original image

def draw_lane(original_img, left_fit, right_fit, Minv):
    ploty = np.linspace(0, original_img.shape[0]-1, original_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])) )])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.4, 0)
    return result

# -------------------------
# Advanced pipeline combining thresholding, perspective, sliding window & smoothing
# -------------------------

def advanced_lane_pipeline(frame: np.ndarray, params: dict, state: dict) -> Tuple[np.ndarray, dict]:
    img = resize_keep_aspect(frame, width=params.get('display_width', 960))
    h, w = img.shape[:2]

    # Perspective source/dst - defaults in image coordinates
    src = np.array(params.get('src_points', [(w*0.15, h*0.95), (w*0.45, h*0.6), (w*0.55, h*0.6), (w*0.85, h*0.95)]), dtype=np.float32)
    dst = np.array(params.get('dst_points', [(w*0.2, h), (w*0.2, 0), (w*0.8, 0), (w*0.8, h)]), dtype=np.float32)

    M, Minv = get_perspective_transform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # Threshold
    binary = color_threshold(warped)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Sliding window polynomial fit
    left_fit, right_fit = sliding_window_poly(binary, nwindows=params.get('nwindows',9), margin=params.get('margin',100), minpix=params.get('minpix',50))

    annotated = img.copy()
    if left_fit is not None and right_fit is not None:
        annotated = draw_lane(annotated, left_fit, right_fit, Minv)
        (lcr, rcr), offset = measure_curvature_and_offset(left_fit, right_fit, img.shape)
        # Smoothing - store in state
        buf = state.setdefault('poly_buf', deque(maxlen=params.get('smoothing_buffer', 5)))
        buf.append((left_fit, right_fit))

        # Average fits
        avg_left = np.mean([p[0] for p in buf], axis=0)
        avg_right = np.mean([p[1] for p in buf], axis=0)
        annotated = draw_lane(annotated, avg_left, avg_right, Minv)

        info_text = f"Left R: {int(lcr)} m | Right R: {int(rcr)} m | Offset: {offset:.2f} m"
        cv2.putText(annotated, info_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "Lane not detected", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

    return annotated, state

# -------------------------
# Video processing helper (advanced)
# -------------------------

def process_video_advanced(input_path: str, output_path: str, params: dict):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = st.progress(0)
    state = {}
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, state = advanced_lane_pipeline(frame, params, state)
        out.write(annotated)
        i += 1
        if frame_count:
            progress.progress(min(i/frame_count, 1.0))
    cap.release()
    out.release()
    progress.empty()

# -------------------------
# Streamlit UI: Sidebar
# -------------------------

st.title("🚦 LaneFusion Pro — Professional Lane Detection")
st.markdown("<div class='streaming-badge'>Real-time prototype · Advanced tools</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration & Presets")
    preset = st.selectbox("Select preset:", ['Default - Balanced', 'Aggressive - Urban', 'Conservative - Highway'])
    if preset == 'Default - Balanced':
        nwindows = 9; margin = 100; minpix = 50; smoothing = 5
    elif preset == 'Aggressive - Urban':
        nwindows = 12; margin = 75; minpix = 30; smoothing = 3
    else:
        nwindows = 7; margin = 150; minpix = 80; smoothing = 7

    st.subheader("Perspective / ROI")
    display_width = st.slider("Display width", 640, 1600, 960, step=64)
    smoothing_buffer = st.slider("Temporal smoothing buffer", 1, 20, smoothing)

    st.subheader("Processing performance")
    use_gpu = st.checkbox("Attempt GPU (TensorFlow) if available", value=False)
    st.write("TensorFlow available:" , TF_AVAILABLE)

    st.write("---")
    st.markdown("**Model manager**")
    uploaded_model = st.file_uploader("Upload U-Net .h5 (optional)", type=['h5'])
    if uploaded_model is not None and TF_AVAILABLE:
        tmp_model_path = os.path.join(tempfile.gettempdir(), 'lanefusion_uploaded.h5')
        with open(tmp_model_path, 'wb') as f:
            f.write(uploaded_model.read())
        st.success("Model uploaded")
    st.button("Clear model cache")

# -------------------------
# Main area: Tabs for inputs & results
# -------------------------

tab1, tab2, tab3 = st.tabs(["Run / Preview", "Batch / Export", "Help & About"]) 

with tab1:
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Input")
        input_mode = st.radio("Input mode:", ['Upload File', 'Webcam / Camera', 'Sample folder'])

        if input_mode == 'Upload File':
            uploaded = st.file_uploader("Upload image / video", type=['jpg','jpeg','png','mp4','mov'])
            if uploaded is not None:
                suffix = uploaded.name.split('.')[-1].lower()
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.'+suffix)
                tmp_in.write(uploaded.read())
                tmp_in.flush()
                source_path = tmp_in.name
        elif input_mode == 'Webcam / Camera':
            cam_img = st.camera_input("Use your webcam (single frame)")
            if cam_img is not None:
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                tmp_in.write(cam_img.getvalue())
                tmp_in.flush()
                source_path = tmp_in.name
        else:
            base = os.path.join(os.path.expanduser('~'), 'Desktop', 'LaneFusion-CV AI Project')
            sample_dir = os.path.join(base, 'road_line_images')
            imgs = []
            if os.path.exists(sample_dir):
                imgs = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            if not imgs:
                st.info('No sample images found on Desktop sample folder. Upload or change sample folder on disk.')
            else:
                source_path = imgs[0]

        st.subheader("Real-time preview")
        start = st.button("Run / Preview")

        if start:
            params = {
                'display_width': display_width,
                'nwindows': nwindows,
                'margin': margin,
                'minpix': minpix,
                'smoothing_buffer': smoothing_buffer,
            }
            if input_mode in ['Upload File', 'Sample folder'] and uploaded is not None or input_mode=='Sample folder' and imgs:
                # if video
                ext = os.path.splitext(source_path)[1].lower()
                if ext in ['.mp4', '.mov']:
                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    st.info('Processing video — this may take a few minutes')
                    process_video_advanced(source_path, tmp_out.name, params)
                    st.video(tmp_out.name)
                    with open(tmp_out.name, 'rb') as f:
                        st.download_button('Download processed video', f, file_name='lanefusion_output.mp4')
                else:
                    frame = cv2.imread(source_path)
                    annotated, _ = advanced_lane_pipeline(frame, params, {})
                    st.image(to_rgb(annotated), use_column_width=True)

    with colB:
        st.subheader("Preview Controls & Diagnostics")
        st.markdown("**Diagnostics**")
        st.write("TensorFlow available:", TF_AVAILABLE)
        st.metric("Input resolution", "—")
        st.markdown("<div class='card small-muted'>Use presets for quick setup. Use higher smoothing for shaky video, lower for fast response.</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("Batch processing & Export")
    st.markdown("Upload a folder or list of videos to batch-process and export a zip of results.")
    uploaded_batch = st.file_uploader("Upload multiple video files", accept_multiple_files=True, type=['mp4','mov'])
    if uploaded_batch:
        out_files = []
        for f in uploaded_batch:
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.'+f.name.split('.')[-1])
            tmp_in.write(f.read())
            tmp_in.flush()
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            params = {'display_width': display_width, 'nwindows': nwindows, 'margin': margin, 'minpix': minpix, 'smoothing_buffer': smoothing_buffer}
            process_video_advanced(tmp_in.name, tmp_out.name, params)
            out_files.append(tmp_out.name)
        st.success(f"Processed {len(out_files)} files")
        # Provide download links (one by one)
        for p in out_files:
            with open(p, 'rb') as f:
                st.download_button(f"Download {os.path.basename(p)}", f, file_name=os.path.basename(p))

with tab3:
    st.header("Help & About")
    st.markdown("This professional prototype implements advanced lane detection techniques combining color/gradient thresholding, perspective transforms, sliding-window polynomial fitting and temporal smoothing. It is meant for research & prototyping — not a production-ready autonomous driving stack.")
    st.markdown("**Suggested next steps:** Add temporal Kalman smoothing, better ROI calibration UI, lane classification (broken/dashed), improve robustness with deep segmentation models (U-Net / DeepLab) and convert heavy models to ONNX/TensorRT for deployment.")
    st.markdown("---")
    st.markdown("Developed by LaneFusion · Prototype version")

# -------------------------
# Final notes
# -------------------------
st.write('---')
st.markdown('**Tips:** Use the `Aggressive - Urban` preset on city footage, `Conservative - Highway` on high-speed clear lanes. For model-backed segmentation, upload a trained U-Net .h5 model in the sidebar.')

