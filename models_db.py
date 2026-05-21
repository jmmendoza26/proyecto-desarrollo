from datetime import datetime
from sqlalchemy import Integer, String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
