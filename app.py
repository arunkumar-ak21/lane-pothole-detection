import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# -----------------------------------------------------------
# IMPORT RL + TELEGRAM MODULES
# -----------------------------------------------------------
from rl_agent import RLAgent
from telegram_notifier import TelegramNotifier

# -----------------------------------------------------------
# MODEL PATHS
# -----------------------------------------------------------
LANE_MODEL_PATH = "best_lane.pt"
POTHOLE_MODEL_PATH = "last.pt"

lane_model = YOLO(LANE_MODEL_PATH)
pothole_model = YOLO(POTHOLE_MODEL_PATH)

# -----------------------------------------------------------
# RL + Telegram Setup
# -----------------------------------------------------------
agent = RLAgent()

notifier = TelegramNotifier(
    bot_token="8541464745:AAEOegMT8BloGTQeJF4fM9UAxLtA1Au0NqE",   # <--- add your token
    chat_id=1872508623,                  # <--- add your chat id
    min_interval_sec=5
)

last_action = 0

# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------
def estimate_distance(box, frame_h):
    x1, y1, x2, y2 = box
    box_h = (y2 - y1) / frame_h
    return max(1, 20 * (1 - box_h))

def rl_reward(action, conf, dist):
    danger = conf > 0.6 and dist < 12
    if danger:
        if action == 2: return 10
        if action == 1: return 5
        return -10
    else:
        if action in (1, 2): return -5
        return 1

# -----------------------------------------------------------
# DETECTION FUNCTIONS
# -----------------------------------------------------------
def detect_lanes(frame):
    results = lane_model(frame)[0]
    return results.plot()

def detect_potholes_with_rl(frame):
    global last_action

    results = pothole_model(frame)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return results.plot()

    best_box = None
    best_conf = 0

    for b in boxes:
        conf = float(b.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = b

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    dist = estimate_distance((x1, y1, x2, y2), frame.shape[0])

    # RL logic
    state = np.array([best_conf, dist, last_action])
    action = agent.choose(state)

    # Send Telegram ONLY for potholes
    if action == 1:
        notifier.send(f"Soft alert: pothole ahead.\nConf={best_conf:.2f}, Dist~{dist:.1f}m")
    elif action == 2:
        notifier.send(f"⚠ STRONG alert: Pothole VERY CLOSE!\nConf={best_conf:.2f}, Dist~{dist:.1f}m")

    # RL update
    reward = rl_reward(action, best_conf, dist)
    next_state = np.array([best_conf, max(dist - 1, 0), action])
    agent.learn(state, action, reward, next_state)

    last_action = action

    return results.plot()

# -----------------------------------------------------------
# LANE + POTHOLE MODE (Lane visual + Pothole RL)
# -----------------------------------------------------------
def detect_both(frame):
    lane_overlay = lane_model(frame)[0].plot()
    final_output = detect_potholes_with_rl(lane_overlay)
    return final_output

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="Lane + Pothole Detection", layout="wide")

st.title("🚗 Real-Time Lane + Pothole Detection with RL + Telegram")
st.markdown("This system detects lanes and potholes. **Alerts are sent ONLY for potholes** using RL.")

# Sidebar
st.sidebar.header("Detection Mode")
mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Lane Only", "Pothole Only (with RL + Telegram)", "Lane + Pothole"]
)

tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📸 Live Camera"])

# -----------------------------------------------------------
# IMAGE UPLOAD
# -----------------------------------------------------------
with tab1:
    st.subheader("Upload Image")
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        if mode == "Lane Only":
            output = detect_lanes(img_np)
        elif mode == "Pothole Only (with RL + Telegram)":
            output = detect_potholes_with_rl(img_np)
        else:
            output = detect_both(img_np)

        st.image(output, caption="Processed Output", use_column_width=True)

# -----------------------------------------------------------
# VIDEO UPLOAD
# -----------------------------------------------------------
with tab2:
    st.subheader("Upload Video")
    vid_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mode == "Lane Only":
                processed = detect_lanes(frame)
            elif mode == "Pothole Only (with RL + Telegram)":
                processed = detect_potholes_with_rl(frame)
            else:
                processed = detect_both(frame)

            stframe.image(processed, channels="RGB")

        cap.release()

# -----------------------------------------------------------
# LIVE CAMERA
# -----------------------------------------------------------
with tab3:
    st.subheader("Live Camera")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not accessible.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mode == "Lane Only":
                processed = detect_lanes(frame)
            elif mode == "Pothole Only (with RL + Telegram)":
                processed = detect_potholes_with_rl(frame)
            else:
                processed = detect_both(frame)

            FRAME_WINDOW.image(processed, channels="RGB")

        cap.release()
