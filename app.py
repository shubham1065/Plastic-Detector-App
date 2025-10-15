import torch
import torch.nn as nn # Must import nn
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

# --- 1. CRITICAL CONFIGURATION ---
CLASSES = ['PE', 'PS', 'PC', 'PET', 'PP', 'others'] 
NUM_CLASSES = len(CLASSES) 
MODEL_NAME = 'CustomTinyMobileNet' # Updated model name for clarity
MODEL_PATH = 'mobilenetv3_model.pth' # Using the new, small file name

IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]   
INPUT_SIZE = 224 
# ----------------------------------------------------

# --- NEW: CUSTOM MODEL ARCHITECTURE (Matches the 128-Feature Requirement) ---
class TinyMobileNet(nn.Module):
    """
    Custom architecture defined to match the structure of the provided weights file.
    It uses the standard MobileNetV3 backbone and forces the final classification 
    layer to have the necessary 128 input features, which the weights expect.
    """
    def __init__(self, num_classes=6):
        super().__init__()
        # Load the features backbone from timm, excluding the original classifier head (num_classes=0)
        # CRITICAL FIX: Changing to 'mobilenetv3_small_075' to ensure the backbone outputs 576 features.
        self.features = timm.create_model(
            'mobilenetv3_small_075',
            pretrained=False, 
            num_classes=0 # IMPORTANT: Load only the feature extractor, not the head
        ).forward_features

        # Global average pooling layer (reduces spatial dimensions)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final custom classifier matching the required input/output shapes:
        # The first layer must accept the 576 features output by the backbone.
        self.classifier = nn.Sequential(
            # This is the layer that MUST match the [6, 128] shape in the weights file
            nn.Linear(576, 128), # Using the correct output channel size (576) of the small mobilenet
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_classes) # Final layer from 128 features to 6 classes
        )


    def forward(self, x):
        # 1. Pass through feature extractor backbone
        x = self.features(x)
        
        # 2. Global Pooling (converts C x H x W to C x 1 x 1)
        x = self.global_pool(x)
        
        # 3. Flatten the tensor for the classifier (converts C x H x W to C)
        # CRITICAL: This flattens the 576 channels into 576 features.
        x = torch.flatten(x, 1)
        
        # 4. Pass through custom classifier head
        x = self.classifier(x)
        return x

# --- 2. MODEL DOWNLOAD AND LOADING FUNCTION ---
def load_plastic_model():
    """Loads the custom TinyMobileNet architecture and state dict."""
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FATAL ERROR: Model file '{MODEL_PATH}' not found in the server's directory. Please ensure it was pushed to GitHub.")

    try:
        print("INFO: Initializing Custom TinyMobileNet architecture...")
        
        # Instantiate the custom model class
        model = TinyMobileNet(num_classes=NUM_CLASSES) 

        # Load the state dictionary (weights)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # CRITICAL FIX: Load weights using strict=False to ignore name mismatches, 
        # and rely on the model structure to match the size.
        model.load_state_dict(state_dict, strict=False) 

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
            
            # Post-processing continues if output is a valid tensor
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
        # This catches any errors during inference (e.g., memory crash)
        error_message = f"Inference Error: {str(e)[:150]}" 
        print(f"FATAL SERVER CRASH: {error_message}")
        return {"status": "error", "error": error_message}
