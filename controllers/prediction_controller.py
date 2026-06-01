from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db
from models.prediction_model import Prediction
from services.inference_service import predict

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_BYTES = 5 * 1024 * 1024
CLASSES = ["Apple", "Banana", "Tomato", "Potato", "Onion"]

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"classes": CLASSES})


@router.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    user_class: str = Form(...),
    db: Session = Depends(get_db)
):
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen JPG o PNG."
        )

    if user_class not in CLASSES:
        raise HTTPException(status_code=400, detail="Clase seleccionada no válida.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="El archivo supera el tamaño máximo permitido de 5 MB."
        )

    result = predict(image_bytes)

    result_value = "Correct" if result["predicted_class"] == user_class else "Incorrect"

    prediction = Prediction(
        filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        user_class=user_class,
        result=result_value,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return templates.TemplateResponse(request, "result.html", {
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"] * 100, 1),
        "user_class": user_class,
        "result": result_value,
    })


@router.get("/history")
def history(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return templates.TemplateResponse(request, "history.html", {
        "predictions": predictions,
        "classes": CLASSES,
    })


@router.delete("/history")
def delete_all(db: Session = Depends(get_db)):
    count = db.query(Prediction).count()
    db.query(Prediction).delete()
    db.execute(text("ALTER SEQUENCE predictions_id_seq RESTART WITH 1"))
    db.commit()
    return JSONResponse({"deleted": count})


@router.delete("/history/{prediction_id}")
def delete_one(prediction_id: int, db: Session = Depends(get_db)):
    p = db.get(Prediction, prediction_id)
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    db.delete(p)
    db.commit()
    return JSONResponse({"deleted": prediction_id})


class PatchUserClass(BaseModel):
    user_class: str


@router.patch("/history/{prediction_id}")
def patch_user_class(prediction_id: int, body: PatchUserClass, db: Session = Depends(get_db)):
    if body.user_class not in CLASSES:
        raise HTTPException(status_code=400, detail="Clase no válida.")
    p = db.get(Prediction, prediction_id)
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    p.user_class = body.user_class
    p.result = "Correct" if p.predicted_class == body.user_class else "Incorrect"
    db.commit()
    db.refresh(p)
    return JSONResponse({"id": p.id, "user_class": p.user_class, "result": p.result})
