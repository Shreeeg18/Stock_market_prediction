from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from utils import fetch_stock_data, preprocess_data, create_input_sequence, predict_price

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-name.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "API running"}


@app.get("/predict/{stock}")
def predict(stock: str):
    try:
        df = fetch_stock_data(stock)
        scaled_data, scaler = preprocess_data(df)
        X = create_input_sequence(scaled_data)
        result = predict_price(model, X, scaler)

        return {
            "stock": stock.upper(),
            "predicted_price": float(result)
        }

    except Exception as e:
        return {"error": str(e)}
