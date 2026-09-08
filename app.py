import os
import io
import base64
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoIdentifyAI")

app = FastAPI(
    title="EcoIdentify AI - Waste Classification API",
    description="Real-time AI-powered Waste Classification & Recycling Guidance API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target Waste Categories
CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Class Emojis
EMOJIS = {
    'Cardboard': '📦',
    'Glass': '🍾',
    'Metal': '🥫',
    'Paper': '📄',
    'Plastic': '🥤',
    'Trash': '🗑️'
}

# Detailed Actionable Recycling & Disposal Recommendations
RECYCLING_GUIDANCE = {
    'Cardboard': {
        'title': 'Cardboard Disposal & Recycling',
        'bin_color': 'Blue Bin / Paper & Cardboard Recycling',
        'bin_badge_color': '#2563eb',
        'steps': [
            'Flatten all cardboard boxes to save space in the recycling bin.',
            'Remove adhesive shipping tape, plastic wrapping, and heavy labels.',
            'Ensure cardboard is dry and free of food grease (greasy pizza boxes go to trash).'
        ],
        'eco_tip': 'Recycling 1 ton of cardboard saves over 46 gallons of oil and 17 trees!'
    },
    'Glass': {
        'title': 'Glass Container Recycling',
        'bin_color': 'Green Bin / Glass Recycling',
        'bin_badge_color': '#16a34a',
        'steps': [
            'Rinse out remaining food or liquid thoroughly with cold water.',
            'Remove metal or plastic caps and lids (recycle them in metal/plastic bins).',
            'Do NOT break glass bottles before placing them in the glass recycling bank.'
        ],
        'eco_tip': 'Glass is 100% recyclable and can be melted and reused endlessly without loss in quality!'
    },
    'Metal': {
        'title': 'Metal & Can Recycling',
        'bin_color': 'Yellow Bin / Metal & Plastics',
        'bin_badge_color': '#eab308',
        'steps': [
            'Empty contents completely and rinse off food or juice residue.',
            'Crush aluminum beverage cans to save space where accepted.',
            'Clean aluminum foil, soda cans, and tin food cans are all 100% recyclable.'
        ],
        'eco_tip': 'Recycling aluminum cans saves 95% of the energy needed to make new ones from raw bauxite ore!'
    },
    'Paper': {
        'title': 'Paper Recycling',
        'bin_color': 'Blue Bin / Mixed Paper Recycling',
        'bin_badge_color': '#2563eb',
        'steps': [
            'Keep paper dry and clean. Remove plastic sleeves or spiral metal wires.',
            'Magazines, newspapers, office paper, and clean paper bags are accepted.',
            'Shredded paper should be placed inside a paper bag before recycling.'
        ],
        'eco_tip': 'Every ton of recycled paper saves 7,000 gallons of water and 4,100 kWh of electricity!'
    },
    'Plastic': {
        'title': 'Plastic Container Recycling',
        'bin_color': 'Yellow Bin / Plastic & Packaging',
        'bin_badge_color': '#eab308',
        'steps': [
            'Rinse container thoroughly to remove food, milk, or liquid residue.',
            'Remove cap/lid if made of non-recyclable material.',
            'Squeeze or crush plastic bottles to save space in yellow bin.'
        ],
        'eco_tip': 'Recycling plastic reduces ocean microplastic pollution and cuts greenhouse gas emissions!'
    },
    'Trash': {
        'title': 'Non-Recyclable General Waste',
        'bin_color': 'Black/Grey Bin / General Landfill Waste',
        'bin_badge_color': '#64748b',
        'steps': [
            'Item is non-recyclable, heavily contaminated, or composite material.',
            'Bag tightly to prevent littering and pest attraction.',
            'Consider reusable alternatives for future purchases to reduce landfill waste.'
        ],
        'eco_tip': 'Try substituting single-use items with reusable glass, silicone, or cloth alternatives!'
    }
}

# Framework & Model Setup
MODEL_ENGINE = "PyTorch-EfficientNetV2"
torch_model = None
torch_transform = None

def init_ai_model():
    global torch_model, torch_transform, MODEL_ENGINE
    try:
        import torch
        import torchvision
        from torchvision import transforms

        logger.info("Initializing PyTorch EfficientNetV2 Deep Learning Model...")
        # Use MobileNetV3 or EfficientNetV2 transfer model for real-time classification
        base_model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        base_model.eval()

        torch_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        torch_model = base_model
        MODEL_ENGINE = "PyTorch-EfficientNetV2"
        logger.info("PyTorch Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize PyTorch model: {str(e)}")

# ImageNet to 6 Waste Class Mapping Matrix
IMAGENET_WASTE_MAP = {
    'Cardboard': [488, 504, 921, 630], # carton, mug/box, book jacket, notebook
    'Glass': [441, 440, 898, 907, 504, 720], # beer bottle, beer glass, wine bottle, bucket, mug, pill bottle
    'Metal': [741, 547, 849, 412, 603, 859], # tin can, can/electric, teapot, ashcan, iron, toaster
    'Paper': [921, 630, 700, 488, 622], # book jacket, notebook, paper towel, carton, envelope
    'Plastic': [737, 898, 720, 907, 441, 911], # pop bottle, water bottle, pill bottle, bucket, plastic bottle, tub
    'Trash': [412, 613, 899, 920, 477] # ashcan, garbage truck, trash, waste, cage
}

def run_ai_inference(pil_img: Image.Image):
    """Run model inference on PIL image and return Top-1 class and probabilities dict."""
    import torch
    import torch.nn.functional as F

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tensor = torch_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = torch_model(tensor)
        probs = F.softmax(logits, dim=1)[0].numpy()

    # Aggregate ImageNet class probabilities into the 6 target waste classes
    waste_scores = {cls: 0.0 for cls in CLASS_NAMES}

    for cls_name, idx_list in IMAGENET_WASTE_MAP.items():
        score = sum(probs[idx] for idx in idx_list if idx < len(probs))
        waste_scores[cls_name] = float(score)

    # Calculate color/texture feature heuristics for fallback stabilization
    img_np = np.array(pil_img.resize((64, 64)), dtype=np.float32) / 255.0
    r_mean, g_mean, b_mean = img_np[:, :, 0].mean(), img_np[:, :, 1].mean(), img_np[:, :, 2].mean()
    gray = img_np.mean(axis=2)
    std_dev = float(np.std(gray))

    # Enhance visual heuristics
    if r_mean > 0.45 and g_mean > 0.35 and b_mean < 0.3: # Brownish -> Cardboard
        waste_scores['Cardboard'] += 0.35
    elif std_dev > 0.22: # Shiny/High Contrast -> Glass/Metal
        waste_scores['Glass'] += 0.20
        waste_scores['Metal'] += 0.20
    elif r_mean > 0.6 and g_mean > 0.6 and b_mean > 0.6: # White/Bright -> Paper
        waste_scores['Paper'] += 0.30

    # Ensure baseline minimum score for all classes
    for cls in CLASS_NAMES:
        waste_scores[cls] += 0.05

    # Normalize to probabilities summing to 1.0 (Softmax style)
    total_score = sum(waste_scores.values())
    final_probs = {cls: round(score / total_score, 4) for cls, score in waste_scores.items()}

    # Determine Top-1
    top_class = max(final_probs, key=final_probs.get)
    confidence = final_probs[top_class]

    return top_class, confidence, final_probs

# Initialize AI model on server start
init_ai_model()

class Base64ImageRequest(BaseModel):
    image: str

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_engine": MODEL_ENGINE,
        "model_loaded": torch_model is not None,
        "classes": CLASS_NAMES
    }

@app.post("/api/predict")
async def predict_base64(payload: Base64ImageRequest):
    if torch_model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded properly.")
    
    try:
        image_data = payload.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        top_class, confidence, probabilities = run_ai_inference(pil_img)
        
        return {
            "success": True,
            "predicted_class": top_class,
            "confidence": confidence,
            "confidence_percentage": f"{round(confidence * 100, 1)}%",
            "emoji": EMOJIS[top_class],
            "probabilities": probabilities,
            "guidance": RECYCLING_GUIDANCE[top_class]
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image processing: {str(e)}")

@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if torch_model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded properly.")
    
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        
        top_class, confidence, probabilities = run_ai_inference(pil_img)
        
        return {
            "success": True,
            "predicted_class": top_class,
            "confidence": confidence,
            "confidence_percentage": f"{round(confidence * 100, 1)}%",
            "emoji": EMOJIS[top_class],
            "probabilities": probabilities,
            "guidance": RECYCLING_GUIDANCE[top_class]
        }
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")

# Mount static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>EcoIdentify AI Backend Running. static/index.html not found.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
