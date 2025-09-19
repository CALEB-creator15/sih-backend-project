# app/models/dummy_model.py

class VehicleDetectionModel:
    def __init__(self):
        # Simulate loading a heavy ML model
        print("Dummy vehicle detection model loaded!")

    def predict(self, data):
        # Simulate prediction output
        return {
            "vehicle_detected": True,
            "vehicle_type": "Car",
            "confidence": 0.87
        }

# Singleton model instance
model_instance = VehicleDetectionModel()
