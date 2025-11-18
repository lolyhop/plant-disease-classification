import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
import yaml
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import PlantModel
from PIL import Image

app = FastAPI(title="Plant Disease Classification API")

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH")
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH")

if CONFIG_PATH is None or WEIGHTS_PATH is None:
    raise RuntimeError(
        "Environment variables MODEL_CONFIG_PATH and MODEL_WEIGHTS_PATH must be set"
    )

with open(Path(CONFIG_PATH)) as f:
    cfg = yaml.safe_load(f)

model = PlantModel(cfg, weights_path=WEIGHTS_PATH)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from uploaded image."""
    image = Image.open(file.file).convert("RGB")
    pred = model.predict(image)
    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
