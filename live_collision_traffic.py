# live_collision_traffic.py
import cv2
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import time

# -----------------------
# CONFIG - replace with real IPs
# -----------------------
BROKER = "10.16.175.58"      # MQTT Broker IP (Device C)
PORT = 1883
TOPIC = "carla/output"

RTSP_URLS = [
    "rtsp://10.16.175.12:8554/cam1",
    "rtsp://10.16.175.12:8554/cam2",
    "rtsp://10.16.175.12:8554/cam3",
    "rtsp://10.16.175.12:8554/cam4"
]

CONF_THRESHOLD = 0.30

# -----------------------
# MQTT setup
# -----------------------
client = mqtt.Client()
client.connect(BROKER, PORT, 60)
client.loop_start()

# -----------------------
# Open RTSP captures
# -----------------------
caps = [cv2.VideoCapture(u) for u in RTSP_URLS]

# -----------------------
# Load YOLO model
# -----------------------
model = YOLO("models/yolov8n.pt")

def is_vehicle_class(name):
    return name in ["car", "truck", "bus", "motorcycle"]

def centroid(box):
    """Calculate centroid of a box [x1, y1, x2, y2]"""
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    return (cx, cy)

# -----------------------
# Initialize per-camera counts
# -----------------------
vehicle_counts = [0] * len(RTSP_URLS)

# Lines positions (pixels)
LINE_TOP_Y = 100
LINE_BOTTOM_Y = 380
LINE_COLOR = (0, 255, 255)

# To store vehicle IDs and last line crossed (simple tracker)
previous_centroids = [{} for _ in RTSP_URLS]  # {centroid_id: last_y}

# Simple centroid ID generator
centroid_id_counter = [0] * len(RTSP_URLS)

# -----------------------
# Main loop
# -----------------------
try:
    while True:
        frames = []

        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            results = model.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD)
            r = results[0]

            boxes = []
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0]) if hasattr(box.cls, "_len_") else int(box.cls)
                    cls_name = model.names.get(cls_id, str(cls_id))
                    if not is_vehicle_class(cls_name):
                        continue
                    xy = box.xyxy[0].tolist()
                    boxes.append(xy)

            # Draw top and bottom lines
            cv2.line(frame, (0, LINE_TOP_Y), (frame.shape[1], LINE_TOP_Y), LINE_COLOR, 2)
            cv2.line(frame, (0, LINE_BOTTOM_Y), (frame.shape[1], LINE_BOTTOM_Y), LINE_COLOR, 2)

            # Count vehicles
            current_centroids = []
            for box in boxes:
                c = centroid(box)
                current_centroids.append(c)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Update counts using simple centroid tracking
            prev_dict = previous_centroids[idx]
            new_prev_dict = {}

            for c in current_centroids:
                assigned = False
                for cid, last_y in prev_dict.items():
                    if abs(c[0] - cid[0]) < 20 and abs(c[1] - last_y) < 50:
                        new_prev_dict[c] = c[1]
                        # Check top line crossing
                        if last_y < LINE_TOP_Y <= c[1]:
                            vehicle_counts[idx] += 1
                        # Check bottom line crossing (leaving)
                        if last_y < LINE_BOTTOM_Y <= c[1]:
                            vehicle_counts[idx] = max(0, vehicle_counts[idx] - 1)
                        assigned = True
                        break
                if not assigned:
                    new_prev_dict[c] = c[1]
            previous_centroids[idx] = new_prev_dict

            # Overlay stats
            cv2.putText(frame, f"Count: {vehicle_counts[idx]}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Coll: N", (6, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Viol: N", (6, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            frames.append(cv2.resize(frame, (640, 480)))

        # Fill missing feeds if < 4
        while len(frames) < 4:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        # Combine into 2x2 grid
        grid = np.vstack((
            np.hstack((frames[0], frames[1])),
            np.hstack((frames[2], frames[3]))
        ))

        cv2.imshow("4 Camera Feeds", grid)

        # Publish MQTT message
        payload = {
            "per_camera_counts": vehicle_counts,
            "per_camera_collision": ["N"]*4,
            "per_camera_violation": ["N"]*4,
            "timestamp": time.time()
        }
        client.publish(TOPIC, json.dumps(payload))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
