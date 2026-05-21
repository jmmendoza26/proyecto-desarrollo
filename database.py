from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# URL de conexión a la base de datos PostgreSQL
DATABASE_URL = "postgresql://postgres:postgres123@localhost:5432/postgres"

# Motor de base de datos
engine = create_engine(DATABASE_URL)

# Fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para los modelos ORM
class Base(DeclarativeBase):
    pass

# Dependencia para inyectar la sesión en las rutas
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
