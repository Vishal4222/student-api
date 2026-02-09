from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="Student Grade Prediction API")

model = torch.load("model.pt", map_location="cpu")
model.eval()

class StudentInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: StudentInput):
    x = torch.tensor([data.features], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).item()
    return {"prediction": float(pred)}
