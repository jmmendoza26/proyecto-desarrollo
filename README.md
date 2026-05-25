# FreshScan

Web app for fruit and vegetable classification using a ResNet18 model served through FastAPI.

Upload a photo, select the correct class, and the model returns the predicted class and confidence score. Every prediction is saved to a PostgreSQL database with a correctness result and is browsable in the history view.

**Recognizes:** Apple · Banana · Tomato · Potato · Onion

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Inference | ResNet18 via Google Colab + ngrok (httpx) |
| Templates | Jinja2 + HTML/CSS |
| Database | PostgreSQL + SQLAlchemy |
| Dataset | [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (Kaggle) |

---

## Project Structure

```
freshscan/
├── main.py          # Routes and app setup
├── model.py         # Inference client (calls Colab via ngrok)
├── database.py      # DB engine and session
├── models_db.py     # Prediction ORM model
├── .env             # COLAB_INFERENCE_URL (not committed)
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
pip install fastapi uvicorn sqlalchemy psycopg2-binary httpx python-dotenv python-multipart pydantic
```

### 3. Configure the inference URL

Create a `.env` file in the project root:

```env
COLAB_INFERENCE_URL=https://your-ngrok-url.ngrok-free.app
```

The model runs on Google Colab and is exposed via ngrok. Start the Colab notebook first, then paste the ngrok URL here before launching the app.

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
| `DELETE` | `/history` | Delete all predictions (resets ID sequence) |
| `DELETE` | `/history/{id}` | Delete a single prediction |
| `PATCH` | `/history/{id}` | Update user-selected class and recompute result |

---

## How It Works

1. Upload a JPG or PNG image (max 5 MB).
2. Select the class you think is correct from the 5 available options — required before submitting.
3. The app sends the image to the ResNet18 model running on Colab.
4. The predicted class is compared against your selection and the result is stored as **Correct** or **Incorrect**.
5. All predictions are visible in the history view, where you can:
   - Edit the selected class per record (result recomputes automatically).
   - Delete individual records.
   - Delete all records at once (ID sequence resets to 1).

---

## Database Schema

Table: `predictions`

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `filename` | String | Uploaded file name |
| `predicted_class` | String | Model output |
| `confidence` | Float | Softmax confidence (0–1) |
| `user_class` | String (nullable) | Class selected by user |
| `result` | String (nullable) | `Correct` or `Incorrect` |
| `created_at` | DateTime | Timestamp |

New columns are added automatically via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on startup — no migration tool required.

---

## Model

ResNet18 pretrained on ImageNet, fine-tuned on Fruits-360 with 5 general classes. `layer4` and the final FC layer (`Linear(512 → 5)`) were unfrozen for fine-tuning; earlier layers remain frozen.

Training used 90 images per class (80 Fruits-360 + 10 real photos). Validation accuracy on Fruits-360: ~99%. On real-world photos: ~70%.

Inference runs remotely on Google Colab via a Flask endpoint exposed through ngrok.

---

## Validation Rules

- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Maximum file size: 5 MB
- A class must be selected before submitting
