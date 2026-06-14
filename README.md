# VisionINK-AI
VisionINK AI is an AI-powered virtual whiteboard that enables users to draw, write, and teach using hand gestures captured through a webcam. Leveraging OpenCV and MediaPipe for real-time hand tracking, the platform provides a touchless digital writing experience with integrated face anonymization for privacy-conscious demonstrations.

---

## ✨ Key Features

**Real-time hand tracking:**
Detects and tracks hand movements with low-latency computer vision.

**Gesture-controlled drawing:**
Enables touchless writing and sketching using finger gestures.

**Multi-color pen system:**
Switch between multiple pen colors through an interactive toolbar.

**Dynamic eraser tool:**
Erase content naturally using pinch gestures or dedicated erase tools.

**Face masking (Pixelate / Blur / Blackout):**
Automatically anonymizes faces using pixelation, blur, or blackout filters.

**Whiteboard export & screenshot saving:**
Provides a clean digital canvas for teaching, brainstorming, and presentations.

**Interactive teaching environment:**
Creates an engaging hands-free platform for demonstrations and instruction.

---

## 🚀 Getting Started

**Prerequisites**

Before running VisionINK AI, ensure you have the following installed:

- Python 3.9+
- Webcam
- Git (optional)

1. Clone the repo
git clone https://github.com/yourusername/VisionINK-AI.git
cd VisionINK-AI

2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies 
pip install -r requirements.txt

4. Run VisionINK AI; python app.py 

---

## Controls

| Action | Gesture / Key |
|---|---|
| Draw | Index Finger Up |
| Move Cursor | Index + Middle Finger Up |
| Erase | Pinch Gesture or Eraser Tool |
| Change Color | Select Toolbar Option |
| Clear Canvas | Clear Tool or `C` Key |
| Save Whiteboard | Save Tool or `S` Key |
| Switch Face Mask Mode | `M` Key |
| Quit Application | `Q` Key |

---

## 🧪 Tech Stack
Python — core logic

Computer vision — OpenCV, MediaPipe

ML and AI — MediaPipe, TensorFlow Lite (via MediaPipe)

Image Processing — NumPy

Face Anonymization - Haar Cascade Face Detection, OpenCV Image Processing

---

# Demo
Below is a snapshot of VisionINK AI's workflow and logic

<img width="1512" height="982" alt="Screenshot 2026-06-14 at 2 05 44 AM" src="https://github.com/user-attachments/assets/2f3b13b6-eddd-4915-9605-f78a04e6f6de" />
