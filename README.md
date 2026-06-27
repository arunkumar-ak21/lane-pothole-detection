# 🚗 Real-Time Lane & Pothole Detection (YOLO + RL + Telegram Alerts)

A real-time road safety monitoring system that detects **lanes** and **potholes** using YOLO models, enhanced with **Reinforcement Learning (RL)** for smarter alert decisions.  
The system supports **image upload**, **video upload**, and **live camera detection**, with **Telegram alerts sent only for potholes**.

---

## 🌐 Live App  
👉 **https://lane-pothole-detection-app.streamlit.app**

---

## ⭐ Features

### 🛣 **Lane Detection**
- YOLO-based lane tracking  
- Works on images, videos, and webcam  

### 🕳️ **Pothole Detection**
- YOLO-based pothole bounding box  
- Distance estimation  
- Confidence calculation  

### 🤖 **Reinforcement Learning (RL)**
- Q-learning agent selects appropriate action:
  - `0` → ignore  
  - `1` → soft alert  
  - `2` → strong alert  
- Learns from confidence + distance  
- Reward shaping reduces false alerts  

### 📲 **Telegram Alert System**
- Sends notifications **only for potholes**  
- Alerts include:
  - confidence  
  - estimated distance  
  - action (soft/strong)  

### 💻 **Streamlit Web App**
- 🚀 Fast UI  
- 📷 Image Upload  
- 🎥 Video Upload  
- 📸 Webcam Support (desktop)  
- 📱 Mobile-friendly camera input (manual capture)  

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Lane Detection | YOLOv11 |
| Pothole Detection | YOLOv11 |
| RL Agent | Q-Learning |
| Backend | Python |
| Web App | Streamlit |
| Alerts | Telegram Bot API |
| CV | OpenCV (headless for deployment) |

---

## 🛠️ Installation (Local Setup)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/abhishektech21/lane-pothole-detection-streamlit.git
cd lane-pothole-detection-streamlit
pip install -r requirements.txt
streamlit run app.py
