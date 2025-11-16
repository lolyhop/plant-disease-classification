import os
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import yaml
from PIL import Image
from model import PlantModel
import uvicorn

app = FastAPI(title="Plant Disease Classification API")

CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH")
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH")

if CONFIG_PATH is None or WEIGHTS_PATH is None:
    raise RuntimeError("Environment variables MODEL_CONFIG_PATH and MODEL_WEIGHTS_PATH must be set")

with open(Path(CONFIG_PATH)) as f:
    cfg = yaml.safe_load(f)

model = PlantModel(cfg, weights_path=WEIGHTS_PATH)

@app.get("/")
def read_root():
    return {"message": "Plant Disease Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from uploaded image."""
    image = Image.open(file.file).convert("RGB")
    pred = model.predict(image)
    return {"prediction": pred}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)