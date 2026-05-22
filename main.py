from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import engine, get_db, Base
from models_db import Prediction
from model import predict

# Extensiones de imagen permitidas
ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}

# Tamaño máximo permitido: 5 MB
MAX_SIZE_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Crear tablas en la base de datos al iniciar
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
    result = predict(image_bytes)

    # Guardar en base de datos
    prediction = Prediction(
        filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"]
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return templates.TemplateResponse(request, "result.html", {
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"] * 100, 1)
    })


@app.get("/history")
def history(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return templates.TemplateResponse(request, "history.html", {
        "predictions": predictions
    })
