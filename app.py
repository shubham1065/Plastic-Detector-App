from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import torch
import timm 
import torch.nn.functional as F
from torchvision import transforms 
from PIL import Image
import io
import os

# --- 1. CONFIGURATION (MUST MATCH KAGGLE NOTEBOOK) ---
# Check your Kaggle notebook for the EXACT classes, mean, and std values!
CLASSES = ['PE', 'PS', 'PC', 'PET', 'PP', 'others'] 
NUM_CLASSES = len(CLASSES) 
MODEL_NAME = 'rexnet_150'
MODEL_PATH = 'C:/Users/hp/Downloads/plastic_best_model.pth'

IMAGENET_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet mean
IMAGENET_STD = [0.229, 0.224, 0.225]   # Standard ImageNet std
INPUT_SIZE = 224 # RexNet-150 standard input size
# ----------------------------------------------------

# --- 2. MODEL LOADING FUNCTION ---
def load_plastic_model():
    """Loads the RexNet model architecture and state dict."""
    try:
        # Create the model using timm
        model = timm.create_model(
            model_name=MODEL_NAME, 
            pretrained=False, 
            num_classes=NUM_CLASSES
        )
        
        # Load the state dictionary (weights) onto the CPU
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # Set the model to evaluation mode (CRITICAL)
        model.eval() 
        return model

    except Exception as e:
        print(f"FATAL ERROR: Model loading failed: {e}")
        # Raise an error to stop the app if the model isn't available
        raise e

# Load the model once when the application starts
plastic_model = load_plastic_model()
app = FastAPI()





from starlette.middleware.cors import CORSMiddleware

# WARNING: Using "*" allows any domain to access your API. 
# This is safe for local testing, but you would replace "*" 
# with your actual domain (e.g., "https://my-website.com") for production.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],
)

# --- END CORS FIX ---
# 1. Mount Static Files to a specific sub-path (/static).
# This prevents the static server from overriding the /predict endpoint.
app.mount("/static", StaticFiles(directory=".", html=False), name="static") 

# 2. Define the Home Route (/) to return the index.html file explicitly.
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error 404: index.html not found!</h1><p>Ensure index.html is in the same directory as app.py.</p>", status_code=404)
                            
def get_preprocess_transform():
    """Defines the PyTorch data transforms (Resizing, Normalization)."""
    return transforms.Compose([
        # 1. Resize: Ensures the shortest side is 256 (for good cropping)
        transforms.Resize(256), 
        # 2. Crop: Centers and crops the image to the model's expected size
        transforms.CenterCrop(INPUT_SIZE), 
        # 3. ToTensor: Converts the PIL image to a Tensor and scales pixel values to [0, 1]
        transforms.ToTensor(), 
        # 4. Normalize: Applies the mean and standard deviation from training
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

preprocess_fn = get_preprocess_transform()

def preprocess_image(image_bytes):
    """Takes image bytes, opens as PIL image, and applies transforms."""
    # 1. Open the image bytes using PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Apply the defined transformations
    tensor = preprocess_fn(image)
    
    # 3. Add a "batch dimension" (the model expects input shape [1, 3, 224, 224])
    return tensor.unsqueeze(0)

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    """API endpoint to receive an image and return plastic waste prediction."""
    
    # 1. Read Image Data
    # FastAPI reads the uploaded file content into memory as bytes
    image_bytes = await file.read()
    
    # 2. Preprocessing
    try:
        # Get the image data ready for the model
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(torch.float32)
    except Exception as e:
        # Return an error if the image file is corrupt or invalid
        print(f"ERROR: Preprocessing crash: {e}")
        return {"error": f"Image preprocessing failed: {e}"}
    
    

    # 3. Inference (Model Prediction)
    plastic_model.eval() # Ensure model is in evaluation mode
    try:
        with torch.no_grad(): # Disable gradient calculation (saves memory and speeds up)
            output = plastic_model(input_tensor)
            
            # Apply Softmax to convert raw outputs (logits) into probabilities
            probabilities = F.softmax(output, dim=1)[0]
            
            # Get the index (position) of the highest probability
            predicted_index = torch.argmax(probabilities).item()
            
    except Exception as e:
        print(f"ERROR: Model inference crash: {e}")
        return {"status": "error", "error": f"Model inference failed: {e}"}

        
    # 4. Post-processing and Result Formatting
    
    # Prepare the confidence score for the website
    confidence_percent = f"{probabilities[predicted_index].item() * 100:.2f}%"
    
    # Prepare all confidences for a detailed output
    all_confidences = {
        CLASSES[i]: f"{probabilities[i].item():.4f}" for i in range(NUM_CLASSES)
    }
    
    # Create the final response JSON
    result = {
        "status": "success",
        "filename": file.filename,
        "predicted_class": CLASSES[predicted_index],
        "confidence": confidence_percent,
        "all_confidences": all_confidences
    }
    print(f"DEBUG: FINAL API RETURN: {result}")
    
    return result

# Simple test endpoint to confirm the API is running
@app.get("/")
def home():
    return {"status": "Plastic Detector API is running", "model": MODEL_NAME}