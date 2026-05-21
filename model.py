from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

# Ruta al archivo .pth en la raíz del proyecto
MODEL_PATH = Path(__file__).parent / "modelo_frutas_vegetales_resnet18.pth"

# Transformaciones idénticas a las usadas en validación durante el entrenamiento
val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model() -> tuple[nn.Module, list[str]]:
    """
    Carga el modelo ResNet18 y las clases desde el archivo .pth.
    Retorna el modelo en modo evaluación y la lista de clases.
    """
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    classes: list[str] = checkpoint["classes"]

    # Reconstruir la misma arquitectura usada en el entrenamiento
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


def predict(image_bytes: bytes, model: nn.Module, classes: list[str]) -> dict:
    """
    Recibe los bytes de una imagen y retorna la clase predicha y la confianza.
    """
    from io import BytesIO

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "predicted_class": classes[pred_idx.item()],
        "confidence": round(confidence.item(), 4)
    }
