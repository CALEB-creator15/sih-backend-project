import json
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ===========================
# MQTT SETTINGS
# ===========================
BROKER = "127.0.0.1"         # Your laptop running the broker
PORT = 1883
CLIENT_ID = "ml_client"
TOPIC_INPUT = "carla/camera_feed"   # Carla -> ML
TOPIC_OUTPUT = "carla/control"      # ML -> Carla

# ===========================
# LOAD YOLOv8 MODEL
# ===========================
# Replace with the path of your exported model
MODEL_PATH = "models/yolov8n.onnx"  # Or yolov8n.tflite / yolov8n.pt
model = YOLO(MODEL_PATH)

# ===========================
# HELPER FUNCTIONS
# ===========================
def decode_image(payload_bytes):
    """Decode bytes into OpenCV image"""
    nparr = np.frombuffer(payload_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def detect_objects(frame):
    """Run YOLO detection on a frame"""
    results = model.predict(frame, imgsz=640)
    detected_objects = {}
    for det in results[0].boxes:
        cls_id = int(det.cls[0].item())
        label = model.names[cls_id]
        detected_objects[label] = detected_objects.get(label, 0) + 1
    return detected_objects

# ===========================
# MQTT CALLBACKS
# ===========================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(TOPIC_INPUT)
    else:
        print("Failed to connect, return code", rc)

def on_message(client, userdata, msg):
    # Receive camera frame from Carla
    frame = decode_image(msg.payload)
    
    # Run detection
    detected_objects = detect_objects(frame)
    
    # Send results back to Carla
    client.publish(TOPIC_OUTPUT, json.dumps(detected_objects))
    print("Detected objects:", detected_objects)

# ===========================
# START MQTT CLIENT
# ===========================
client = mqtt.Client(client_id=CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
client.loop_start()  # Background thread

print("Live Carla ML client running. Waiting for camera frames...")

# Keep the script alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping client...")
    client.loop_stop()
    client.disconnect()
