import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# URL del servidor de inferencia en Colab (via ngrok)
COLAB_URL = os.getenv("COLAB_INFERENCE_URL", "").rstrip("/")

def predict(image_bytes: bytes) -> dict:
    """
    Envía la imagen al servidor de inferencia en Colab
    y retorna la clase predicha y la confianza.
    """
    if not COLAB_URL:
        raise RuntimeError(
            "COLAB_INFERENCE_URL no está definida en el archivo .env"
        )

    response = httpx.post(
        f"{COLAB_URL}/predict",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=30.0
    )

    response.raise_for_status()
    return response.json()
