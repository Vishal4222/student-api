from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI(title="Student Grade Prediction API")

MODEL_FILE = "model_scripted.pt"
SIZE_FILE = "input_size.txt"

# Load model once at startup
model = torch.jit.load(MODEL_FILE, map_location="cpu")
model.eval()

# Load expected input size
with open(SIZE_FILE, "r") as f:
    INPUT_SIZE = int(f.read().strip())

class StudentInput(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "input_size": INPUT_SIZE}

@app.post("/predict")
def predict(data: StudentInput):
    if len(data.features) != INPUT_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature length. Expected {INPUT_SIZE} values but got {len(data.features)}."
        )

    x = torch.tensor([data.features], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).item()

    return {"prediction": float(pred)}
