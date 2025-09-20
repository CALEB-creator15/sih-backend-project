# app/routers/traffic.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

router = APIRouter()

# ---- In-memory stores (temporary; later replace with DB) ----
sensor_store = {}        # key: sensor_id -> last reading (dict)
incident_store: List[dict] = []
light_store = {}         # key: junction_id -> last control settings

# ---- Pydantic models ----
class SensorData(BaseModel):
    sensor_id: str
    vehicle_count: int = Field(..., ge=0)
    avg_speed_kmph: Optional[float] = None
    density: Optional[float] = None
    timestamp: Optional[datetime] = None

class LightControl(BaseModel):
    junction_id: str
    green_duration_sec: int = Field(..., ge=0)
    amber_duration_sec: Optional[int] = Field(default=5, ge=0)
    red_duration_sec: Optional[int] = Field(default=30, ge=0)

class IncidentReport(BaseModel):
    location: str
    incident_type: str
    severity: int = Field(..., ge=1, le=5)
    description: Optional[str] = None
    reported_at: Optional[datetime] = None


# ---- Endpoints ----
@router.post("/sensor", status_code=201)
async def post_sensor(data: SensorData):
    # Save last reading for sensor
    payload = data.dict()
    payload["received_at"] = datetime.utcnow().isoformat()
    sensor_store[data.sensor_id] = payload

    # simple logic example: mark congestion if vehicle_count > threshold
    congestion = data.vehicle_count > 50 or (data.density is not None and data.density > 0.8)

    return {
        "status": "received",
        "sensor_id": data.sensor_id,
        "congestion": congestion,
        "stored_at": payload["received_at"]
    }

@router.get("/sensor/{sensor_id}")
async def get_sensor(sensor_id: str):
    record = sensor_store.get(sensor_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sensor not found")
    return record

@router.post("/light", status_code=200)
async def update_light(control: LightControl):
    payload = control.dict()
    payload["updated_at"] = datetime.utcnow().isoformat()
    light_store[control.junction_id] = payload
    # placeholder response — in the future we will send commands to controllers
    return {
        "status": "updated",
        "junction_id": control.junction_id,
        "green_duration_sec": control.green_duration_sec,
        "updated_at": payload["updated_at"]
    }

@router.post("/incident", status_code=201)
async def report_incident(incident: IncidentReport):
    payload = incident.dict()
    payload["reported_at"] = (incident.reported_at or datetime.utcnow()).isoformat()
    payload["id"] = len(incident_store) + 1
    incident_store.append(payload)
    return {"status": "reported", "incident_id": payload["id"], "reported_at": payload["reported_at"]}

@router.get("/incidents")
async def list_incidents():
    return incident_store
