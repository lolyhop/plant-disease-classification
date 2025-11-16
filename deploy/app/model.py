import torch
from pathlib import Path
from typing import Any, Dict
from PIL import Image
from src.models import cnn, mlp, t2t_vit, vit
from torchvision import transforms

MODEL_REGISTRY: Dict[str, Any] = {
    "vit_2021_orig": vit.VisionTransformer,
    "t2t_vit": t2t_vit.T2TViT,
    "mlp": mlp.MLP,
    "deep_mlp": mlp.DeepMLP,
    "resnet": cnn.ResNet,
    "densenet": cnn.DenseNet,
    "efficientnet": cnn.EfficientNet,
}

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def preprocess_image(img: Image.Image, image_size: int) -> torch.Tensor:
    """Preprocess a single image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


class PlantModel:
    def __init__(self, cfg: Dict, weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ModelClass = MODEL_REGISTRY[cfg["model"]["name"]]
        self.model = ModelClass(**cfg["model"]["params"]).to(self.device)

        if weights_path is None:
            weights_path = Path(cfg["training"]["log_dir"]) / "best_model.pt"
        else:
            weights_path = Path(weights_path)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.cfg = cfg

    def predict(self, img: Image.Image) -> int:
        tensor = (
            preprocess_image(img, self.cfg["data"]["image_size"])
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}
