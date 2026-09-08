# ♻️ EcoIdentify AI - Real-Time AI Waste Classifier

**EcoIdentify AI** is an end-to-end, high-performance web application designed to classify waste materials in real time using Computer Vision & Deep Learning (PyTorch EfficientNetV2). It features a modern dark-mode glassmorphic interface, WebRTC live camera streaming with target reticles, auto-scanning, drag-and-drop file uploading, class probability meters, and actionable recycling guidance.

---

## 🌟 Key Features

1. **Dual Input Mode**:
   - 📷 **Live Camera Stream**: Real-time video feed via WebRTC with animated scanner frame, reticles, "Capture & Classify" button, and "Continuous Auto-Scan" mode (scans every 1.5s).
   - ☁️ **Upload Image Mode**: Drag-and-drop file uploader supporting `.jpg`, `.png`, and `.webp` images with instant preview and classification.

2. **Real-time AI Prediction**:
   - **Top-1 Category Display**: Displays predicted waste category emoji, category name, and confidence percentage badge.
   - **Class Probability Visualizer**: Interactive horizontal progress bars showing probability distribution across all 6 waste classes (`Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`).

3. **Disposal & Recycling Guidance Panel**:
   - Actionable step-by-step recycling recommendations based on the predicted class.
   - Bin color tag badges (e.g., Blue Bin for Cardboard/Paper, Yellow Bin for Plastic/Metal, Green Bin for Glass, Black/Grey Bin for Trash).
   - Environmental eco-tips highlighting energy and resource savings.

4. **Modern UI Design System**:
   - Dark slate palette (`#0f172a`), emerald accents (`#10b981`), glassmorphic containers (`backdrop-filter: blur(16px)`), responsive grid layout for desktop and mobile.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyTorch, Torchvision, PIL (Pillow), NumPy, OpenCV.
- **Frontend**: HTML5, CSS3 (Vanilla CSS with Flexbox/Grid & Glassmorphism), Modern JavaScript (WebRTC MediaDevices, Fetch API, HTML5 Canvas).
- **AI Engine**: PyTorch EfficientNetV2 Deep Learning Model.

---

## 📁 Project Directory Structure

```
wms/
├── app.py                      # FastAPI web server & AI inference engine
├── requirements.txt            # Python dependency list
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation & setup guide
├── static/
│   ├── index.html              # Responsive HTML5 single-page interface
│   ├── style.css               # Modern dark-mode glassmorphic stylesheet
│   └── camera.js               # WebRTC camera handler, frame canvas, drag-and-drop, API communication
└── model/
    └── best_model_finetuned224.keras # Pre-trained Keras/Deep Learning model weights
```

---

## 🚀 Quick Start Guide

### Step 1: Clone or Navigate to the Workspace
Open PowerShell or Terminal and navigate to the project directory:
```powershell
cd c:\Users\Binoy\Desktop\wms
```

### Step 2: Install Dependencies
Install all required Python packages using `pip`:
```powershell
pip install -r requirements.txt
```

> [!NOTE]
> The dependencies use `torch` (PyTorch), which supports Python 3.10 through Python 3.14.

### Step 3: Run the Server
Launch the FastAPI application server:
```powershell
python app.py
```
*Or using uvicorn directly:*
```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4: Open in Web Browser
Open your favorite web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🏷️ Target Waste Categories

| Category | Emoji | Disposal Bin | Key Action Step |
| :--- | :---: | :--- | :--- |
| **Cardboard** | 📦 | Blue Bin | Flatten boxes, remove shipping tape, keep dry and grease-free. |
| **Glass** | 🍾 | Green Bin | Rinse thoroughly, remove metal/plastic lids, do not break glass. |
| **Metal** | 🥫 | Yellow Bin | Empty and rinse cans, crush beverage cans to save bin space. |
| **Paper** | 📄 | Blue Bin | Keep dry, remove plastic sleeves or spiral metal wires. |
| **Plastic** | 🥤 | Yellow Bin | Rinse thoroughly, remove caps, crush bottles to save space. |
| **Trash** | 🗑️ | Black/Grey Bin | Non-recyclable or contaminated waste. Bag tightly. |

---

## 📡 REST API Reference

### `GET /api/health`
Check model engine and server health status.
**Response**:
```json
{
  "status": "healthy",
  "model_engine": "PyTorch-EfficientNetV2",
  "model_loaded": true,
  "classes": ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
}
```

### `POST /api/predict`
Classify an image sent as base64 string.
**Payload**:
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### `POST /api/predict-file`
Classify an image uploaded as `multipart/form-data` file.

---

## 📜 License
Licensed under the [MIT License](LICENSE). Built for sustainable waste sorting and environmental intelligence.
