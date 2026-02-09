from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="Student Grade Prediction API")

MODEL_FILE = "model_state.pt"  # make sure this file exists in your repo

class StudentInput(BaseModel):
    features: list[float]

def build_model(input_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

# Load state_dict once at startup and infer input size safely
try:
    state = torch.load(MODEL_FILE, map_location="cpu")

    # infer input size from first linear layer weights
    # for Sequential: first layer is index 0 -> "0.weight"
    first_w = state.get("0.weight", None)
    if first_w is None:
        raise ValueError("Could not find '0.weight' in state_dict. Make sure you saved a Sequential model state_dict.")

    INPUT_SIZE = first_w.shape[1]

    model = build_model(INPUT_SIZE)
    model.load_state_dict(state)
    model.eval()

except Exception as e:
    # If this fails, Render logs will show the real reason
    raise RuntimeError(f"Model load failed: {e}")

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
