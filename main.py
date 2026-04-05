from fastapi import FastAPI
import joblib

# Initialize app
app = FastAPI()

# Load model once (important)
model = joblib.load("model.pkl")

# Health check route
@app.get("/")
def home():
    return {"message": "API is running"}

# Prediction route
@app.post("/predict")
def predict(data: dict):
    features = data["features"]
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}