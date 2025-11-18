# Plant Disease Classification 🌿

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red) ![License](https://img.shields.io/badge/license-MIT-green)

This project implements deep learning models for **plant disease classification** using leaf images. We experiment with several model architectures:

* Multi-Layer Perceptron (MLP)
* Convolutional Neural Networks (CNN)
* Vision Transformers (ViT)

![Project Overview](docs/imgs/intro_v2.png)

---

## Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/lolyhop/plant-disease-classification.git
cd plant-disease-classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

```bash
bash scripts/download_raw_data.sh
```

---

## Repository Structure

```
.
├── configs/                  # Model configurations
├── data/                     # Dataset and logs
├── docs/                     # Images and documentation
├── front/                    # Streamlit frontend
├── notebooks/                # Jupyter notebooks for exploration
├── scripts/                  # Helper scripts
├── src/                      # Code for models, training, and utilities
├── train.py                  # Training script
├── inference.py              # Inference script
├── requirements.txt
└── README.md
```

---

## Run Experiments

You can run training and inference for any model using a config file.

**Training:**

```bash
python train.py --config configs/<config_name>.yaml
```

Examples:

```bash
python train.py --config configs/resnet.yaml
python train.py --config configs/efficientnet.yaml
python train.py --config configs/vit_2021.yaml
```

**Inference:**

```bash
python inference.py --config configs/<config_name>.yaml
```

Examples:

```bash
python inference.py --config configs/mlp.yaml
python inference.py --config configs/densenet.yaml
python inference.py --config configs/t2t_vit.yaml
```

This approach works for all models and keeps commands consistent.

---

## Model API Deployment

### 1. Build the Docker image

```bash
cd plant-disease-classification
docker build -f deploy/Dockerfile -t plant-classifier .
```

### 2. Run the API container

Mount your local `data` directory and provide model config and weights:

```bash
cd plant-disease-classification
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e MODEL_CONFIG_PATH=/app/configs/<your_model>.yaml \
  -e MODEL_WEIGHTS_PATH=/app/data/logs/<your_model>/best_model.pt \
  plant-classifier
```

* `-p 8000:8000` exposes the API on port 8000.
* `-v $(pwd)/data:/app/data` allows the container to access model weights.
* `MODEL_CONFIG_PATH` and `MODEL_WEIGHTS_PATH` must point to the YAML config and `.pt` file **inside the container**.

### 3. Test the API

OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

Send a POST request to `/predict` with a leaf image:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path_to_leaf_image.jpg"
```

The API returns a JSON dictionary with class probabilities:

```json
{
  "Tomato___Early_blight": 0.92,
  "Tomato___Leaf_Mold": 0.05,
  "Tomato___healthy": 0.02,
  ...
}
```

---

## Run Frontend

Navigate to the frontend directory and set environment variables:

```bash
cd front/app

export MODEL_CONFIG_PATH=../../configs/<config_name>.yaml
export MODEL_WEIGHTS_PATH="weights/best_model.pt"

python main.py
```

Available configs: `deep_mlp.yaml`, `densenet.yaml`, `efficientnet.yaml`, `mlp.yaml`, `resnet.yaml`, `t2t_vit.yaml`, `vit_2021.yaml`

---

## Model Performance

| Model                         | Accuracy | Precision | Recall | F1-score |
| ----------------------------- | -------- | --------- | ------ | -------- |
| DenseNet-121                  | 0.9961   | 0.9962    | 0.9960 | 0.9960   |
| ResNet-18                     | 0.9959   | 0.9958    | 0.9959 | 0.9958   |
| EfficientNet-B0               | 0.9950   | 0.9952    | 0.9947 | 0.9948   |
| Tokens-to-Token ViT (T2T-ViT) | 0.9929   | 0.9930    | 0.9929 | 0.9929   |
| Vision Transformer (ViT)      | 0.9909   | 0.9911    | 0.9908 | 0.9909   |
| Deep MLP                      | 0.7620   | 0.7755    | 0.7618 | 0.7601   |
| MLP                           | 0.0552   | 0.0430    | 0.0522 | 0.0263   |

Top performing models are CNNs and Transformer-based models. Simple MLPs perform poorly, highlighting the importance of spatial feature extraction.

---

## Tech Stack

* Python 3.10+
* PyTorch
* Albumentations
* Torchvision
