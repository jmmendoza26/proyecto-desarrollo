# FreshScan

Web app for fruit and vegetable classification using a ResNet18 model served through FastAPI.

Upload a photo and the model returns the predicted class and confidence score. Every prediction is saved to a PostgreSQL database and browsable in the history view.

**Recognizes:** Apple · Banana · Tomato · Potato · Onion

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Inference | PyTorch + ResNet18 (transfer learning) |
| Templates | Jinja2 + HTML/CSS |
| Database | PostgreSQL + SQLAlchemy |
| Dataset | [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (Kaggle) |

---

## Project Structure

```
freshscan/
├── main.py                                   # Routes and app setup
├── model.py                                  # Model loading and inference
├── database.py                               # DB engine and session
├── models_db.py                              # Prediction ORM model
├── modelo_frutas_vegetales_resnet18.pth      # Trained model weights
├── static/
│   └── css/
│       └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    └── history.html
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/freshscan.git
cd freshscan
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary torch torchvision pillow python-multipart
```

### 3. Add the model file

Place `modelo_frutas_vegetales_resnet18.pth` in the project root. The file is not included in the repository due to its size.

### 4. Create the PostgreSQL database

```bash
psql -U postgres -c "CREATE DATABASE freshscan;"
```

Or create it through pgAdmin. Then update the connection URL in `database.py`:

```python
DATABASE_URL = "postgresql://postgres:your_password@localhost:5432/freshscan"
```

### 5. Run the app

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Upload form |
| `POST` | `/predict` | Run inference and save result |
| `GET` | `/history` | All past predictions |

---

## Model

ResNet18 pretrained on ImageNet, fine-tuned on the Fruits-360 dataset with 5 general classes. The final fully connected layer was replaced to output 5 classes. Only the FC layer was trained (all other weights frozen).

Training used 50 images per class (40 train / 10 validation). On the Fruits-360 validation set, the model reaches ~98% accuracy. On real-world photos the accuracy drops to approximately 66%.

The `.pth` file stores both the model weights and the class list:

```python
{
    "model_state_dict": ...,
    "classes": ["Apple", "Banana", "Tomato", "Potato", "Onion"]
}
```

---

## Validation Rules

- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Maximum file size: 5 MB

---

## Notes

- The database tables are created automatically on first run via SQLAlchemy's `create_all`.
- Model weights load on startup using FastAPI's `lifespan` context.
- Predictions in the history view are sorted newest first.
