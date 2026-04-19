import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms, models
import io

app = FastAPI()

# Enable CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# Load model
model = models.efficientnet_b4(weights=None)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(model.classifier[1].in_features, 2)
)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    result = "AI Generated" if pred.item() == 0 else "Real Image"

    return {"prediction": result}