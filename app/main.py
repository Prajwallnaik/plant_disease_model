from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnet_b3_tomato.pth")
CLASS_NAMES = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]
IMAGE_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
MODEL = None

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        print(f"Loading PyTorch model from {MODEL_PATH}...")
        try:
            # Recreate architecture
            try:
                model = models.efficientnet_b3(weights=None)
            except:
                model = models.efficientnet_b3(pretrained=False) # Fallback
                
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_features, len(CLASS_NAMES))
            )
            
            # Load weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            MODEL = model
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Predictions will fail until model is trained.")

load_model()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Define Transforms (Same as Validation)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_file_as_image(data) -> torch.Tensor:
    image = Image.open(BytesIO(data)).convert('RGB')
    tensor = transform(image).unsqueeze(0) # Add batch dimension
    return tensor.to(DEVICE)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if MODEL is None:
        # Try loading again just in case it was trained after startup
        load_model()
        if MODEL is None:
            return {"error": "Model not loaded. Please train the model first."}
    
    try:
        image_tensor = read_file_as_image(await file.read())
        
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence.item())
        }
    except Exception as e:
        return {"error": str(e)}

# Mount static files
if os.path.exists("app/static"):
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
elif os.path.exists("static"):
     app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
