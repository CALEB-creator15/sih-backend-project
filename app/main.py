from fastapi import FastAPI
from pydantic import BaseModel
from app.models.dummy_model import model_instance
from app.routers.traffic import router as traffic_router
# Create the FastAPI app instance
app = FastAPI(
    title="Intelligent Traffic Management System Backend",
    description="Backend API for traffic monitoring, ML inference, and IoT data handling",
    version="0.1"
)
app.include_router(traffic_router, prefix="/traffic", tags=["traffic"])

# Root endpoint (for quick testing)
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI is working!"}

# Simulated ML model load endpoint
@app.get("/load_model")
async def load_model():
    # In future you'll load your actual ML model here.
    return {"status": "Model loaded successfully!"}

# -------- Step 3: Model inference --------
class VehicleData(BaseModel):
    # For now, simple placeholder (later: image base64 or sensor JSON)
    sensor_value: float

@app.post("/predict_vehicle")
async def predict_vehicle(data: VehicleData):
    result = model_instance.predict(data)
    return {"prediction": result}
