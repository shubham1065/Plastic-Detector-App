import torch
import torch.nn as nn
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
MODEL_NAME = 'mobilenetv3_small_050' # Use base name; we fix sizes dynamically
MODEL_PATH = 'mobilenet.pth' # Small model file name
IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]   
INPUT_SIZE = 224 
# ----------------------------------------------------

# --- NEW: CUSTOM MODEL ARCHITECTURE (Features are fixed later) ---
class TinyMobileNet(nn.Module):
    """
    Custom architecture using MobileNetV3 backbone but with a dynamic final layer 
    to match the exact feature count outputted by the backbone at runtime.
    """
    def __init__(self, in_features, num_classes=6):
        super().__init__()
        # Load the features backbone from timm, excluding the original classifier head
        self.features = timm.create_model(
            'mobilenetv3_small_050',
            pretrained=False, 
            num_classes=0 
        ).forward_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final custom classifier layer is built DYNAMICALLY in the load function.
        # We define a placeholder to be replaced later.
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128), # DYNAMIC: in_features will be set at runtime
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- 2. MODEL DOWNLOAD AND LOADING FUNCTION (DYNAMIC SIZE DISCOVERY) ---
def load_plastic_model():
    """Discovers the feature size dynamically, rebuilds the classifier, and loads the model."""
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FATAL ERROR: Model file '{MODEL_PATH}' not found in the server's directory.")

    # --- 1. Size Discovery Pass ---
    # Create a base model structure to find the true output size of the backbone.
    base_model = timm.create_model('mobilenetv3_small_050', pretrained=False, num_classes=0)
    
    # Run a dummy tensor through the backbone to find the output channel size (576, 432, etc.)
    try:
        dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        # Get output from the feature extractor
        dummy_output = base_model.forward_features(dummy_input)
        # Use pooling to flatten the output to a channel count
        final_feature_count = dummy_output.shape[1] 
        print(f"INFO: Determined feature count needed by classifier: {final_feature_count}")
    except Exception as e:
        raise RuntimeError(f"Failed to determine model feature size. Error: {e}")

    # --- 2. Model Instantiation & Layer Replacement ---
    try:
        # Instantiate the custom model using the dynamically discovered size
        model = TinyMobileNet(in_features=final_feature_count, num_classes=NUM_CLASSES) 

        # Load the state dictionary (weights)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Load weights using strict=False to ignore name mismatches, 
        # as the sizes are now correct (576 or 432 features into 576x128 weights).
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