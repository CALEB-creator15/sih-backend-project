from fastapi import FastAPI

# Create the FastAPI app instance
app = FastAPI(
    title="Intelligent Traffic Management System Backend",
    description="Backend API for traffic monitoring, ML inference, and IoT data handling",
    version="0.1"
)

# Root endpoint (for quick testing)
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI is working!"}

# Simulated ML model load endpoint
@app.get("/load_model")
async def load_model():
    # In future you'll load your actual ML model here.
    return {"status": "Model loaded successfully!"}
