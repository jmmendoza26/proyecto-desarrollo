from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import engine, get_db, Base
from models_db import Prediction
from model import load_model, predict

# Extensiones de imagen permitidas
ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}

# Tamaño máximo permitido: 5 MB
MAX_SIZE_BYTES = 5 * 1024 * 1024

# Variables globales para el modelo y las clases
ml_model = None
ml_classes = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Crear tablas en la base de datos al iniciar
    Base.metadata.create_all(bind=engine)

    # Cargar el modelo al iniciar
    global ml_model, ml_classes
    ml_model, ml_classes = load_model()

    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validar tipo de archivo
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen JPG o PNG."
        )

    # Leer bytes y validar tamaño
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="El archivo supera el tamaño máximo permitido de 5 MB."
        )

    # Realizar predicción
    result = predict(image_bytes, ml_model, ml_classes)

    # Guardar en base de datos
    prediction = Prediction(
        filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"]
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"] * 100, 1)
    })


@app.get("/history")
def history(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return templates.TemplateResponse("history.html", {
        "request": request,
        "predictions": predictions
    })
