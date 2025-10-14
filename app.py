import torch
import timm 
import torch.nn.functional as F
from torchvision import transforms 
from PIL import Image
import io
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

CLASSES = ['PE', 'PS', 'PC', 'PET', 'PP', 'others'] 
NUM_CLASSES = len(CLASSES) 

# Changed model name from 'rexnet_150' to a memory-efficient MobileNet
MODEL_NAME = 'mobilenetv2_100' 
MODEL_PATH = 'mobilenetv3_model.pth'

IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]   
INPUT_SIZE = 224 
# ----------------------------------------------------

# --- 2. MODEL LOADING FUNCTION ---
def load_plastic_model():
    """Loads the MobileNet architecture and state dict directly from the repository."""
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FATAL ERROR: Model file '{MODEL_PATH}' not found in the server's directory. Please ensure it was pushed to GitHub.")

    try:
        print("INFO: Initializing MobileNet model architecture...")
        
        # Create the MobileNet architecture
        model = timm.create_model(
            model_name=MODEL_NAME, 
            pretrained=False, 
            num_classes=NUM_CLASSES
        )
        
        # Load the weights onto the CPU (This step is now fast and uses minimal memory)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        model.eval() 
        return model

    except Exception as e:
        print(f"FATAL ERROR: Model loading failed: {e}")
        raise e

# Load the model once when the application starts
plastic_model = load_plastic_model()
app = FastAPI()

# --- CORS MIDDLEWARE (REQUIRED FOR WEB APP INTERACTION) ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- STATIC FILE SERVING AND HOME ROUTE ---
app.mount("/static", StaticFiles(directory=".", html=False), name="static") 

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error 404: index.html not found!</h1><p>Ensure index.html is in the same directory as app.py.</p>", status_code=404)

# --- PREPROCESSING FUNCTIONS ---
def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(INPUT_SIZE), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

preprocess_fn = get_preprocess_transform()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess_fn(image)
    return tensor.unsqueeze(0)

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    
    # 1. Read Image Data
    image_bytes = await file.read()
    
    # 2. Preprocessing
    try:
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(torch.float32)
    except Exception as e:
        print(f"ERROR: Preprocessing crash: {e}")
        return {"error": f"Image preprocessing failed: {e}"}
    
    # 3 & 4. INFERENCE AND POST-PROCESSING
    plastic_model.eval() 
    
    try: 
        with torch.no_grad(): 
            output = plastic_model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_index = torch.argmax(probabilities).item()
            
        # Post-processing
        confidence_percent = f"{probabilities[predicted_index].item() * 100:.2f}%"
        all_confidences = {
            CLASSES[i]: f"{probabilities[i].item():.4f}" for i in range(NUM_CLASSES)
        }
        
        result = {
            "status": "success",
            "filename": file.filename,
            "predicted_class": CLASSES[predicted_index],
            "confidence": confidence_percent,
            "all_confidences": all_confidences
        }
        
        return result

    except Exception as e:
        print(f"ERROR: Model inference crash: {e}")
        return {"status": "error", "error": f"Model inference failed: {e}"}



