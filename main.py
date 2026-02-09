from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="Student Grade Prediction API")

# IMPORTANT: set this to the number of features your model was trained with
# We'll compute it from the first request if you prefer, but best is to set it manually.
INPUT_SIZE = None

# Define the SAME architecture you used during training
def build_model(input_size: int):
    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

class StudentInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: StudentInput):
    global INPUT_SIZE

    # set input size from request the first time
    if INPUT_SIZE is None:
        INPUT_SIZE = len(data.features)

    model = build_model(INPUT_SIZE)
    state = torch.load("model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor([data.features], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).item()

    return {"prediction": float(pred)}

