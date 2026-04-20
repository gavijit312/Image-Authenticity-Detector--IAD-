import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from torchvision import transforms, models

app = FastAPI()

# Enable CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["AI Generated", "Real Image"]

# Load model
model = models.efficientnet_b4(weights=None)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(model.classifier[1].in_features, 2)
)

candidate_paths = [
    BASE_DIR / "best_model.pth",
    BASE_DIR.parent / "best_model.pth",
]

selected_model_path = None
last_error = None

for path in candidate_paths:
    if not path.exists() or path.stat().st_size == 0:
        continue
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        selected_model_path = path
        break
    except Exception as exc:
        last_error = exc

if selected_model_path is None:
    if last_error is not None:
        raise RuntimeError(
            "Could not load a valid model checkpoint. "
            f"Checked: {candidate_paths}. Last error: {type(last_error).__name__}: {last_error}"
        )
    raise FileNotFoundError(
        "No non-empty model checkpoint found. "
        f"Checked: {candidate_paths}"
    )

model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {
        "message": "API is running",
        "device": str(DEVICE),
        "model_path": str(selected_model_path),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    # Test-time augmentation helps stabilize prediction confidence.
    tta_images = [
        transform(image),
        transform(ImageOps.mirror(image)),
    ]
    img = torch.stack(tta_images).to(DEVICE)

    with torch.inference_mode():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        mean_probs = probs.mean(dim=0, keepdim=True)
        conf, pred = torch.max(mean_probs, 1)

    pred_idx = pred.item()
    confidence = float(conf.item())
    probabilities = {
        CLASS_NAMES[idx]: float(prob.item())
        for idx, prob in enumerate(mean_probs[0])
    }

    result = CLASS_NAMES[pred_idx]

    return {
        "prediction": result,
        "class_id": pred_idx,
        "confidence": round(confidence, 6),
        "probabilities": probabilities,
        "filename": file.filename,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)