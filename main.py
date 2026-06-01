from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from database import engine, Base
from controllers.prediction_controller import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text(
            "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS user_class VARCHAR"
        ))
        conn.execute(text(
            "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS result VARCHAR"
        ))
        conn.commit()
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)
