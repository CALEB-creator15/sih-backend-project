import cv2
import numpy as np
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json

# -----------------------
# MQTT setup (publish to Device C)
# -----------------------
BROKER = "10.16.175.58"  # Replace with Device C IP
PORT = 1883
TOPIC = "carla/output"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)
client.loop_start()

# -----------------------
# RTSP feeds from Device A (4 cameras)
# -----------------------
rtsp_urls = [
    "rtsp://10.16.175.12:8554/cam1",
    "rtsp://10.16.175.12:8554/cam2",
    "rtsp://10.16.175.12:8554/cam3",
    "rtsp://10.16.175.12:8554/cam4"
]

caps = [cv2.VideoCapture(url) for url in rtsp_urls]

# -----------------------
# YOLOv8 model
# -----------------------
model = YOLO("models/yolov8n.pt")  # Ensure model exists

# -----------------------
# Vehicle counting setup
# -----------------------
entry_line = 200
exit_line = 400

# -----------------------
# Helper functions
# -----------------------
def centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2)/2), int((y1 + y2)/2)

def check_collision(box1, box2):
    c1 = centroid(box1)
    c2 = centroid(box2)
    distance = np.linalg.norm(np.array(c1)-np.array(c2))
    threshold = 30  # decreased distance for higher accuracy
    return distance < threshold

def is_vehicle_class(cls_name):
    return cls_name in ["car", "truck", "bus", "motorcycle"]

def is_violation(cls_name, box):
    # placeholder for basic violation logic
    return False

# -----------------------
# Main loop
# -----------------------
while True:
    frames = []
    total_vehicles = 0
    collision_flag = False
    violation_flag = False

    # Process each camera feed
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480,640,3), dtype=np.uint8)

        results = model.predict(source=frame, imgsz=640, conf=0.3)[0]

        boxes = []
        labels = []
        vehicles_in_region = 0

        # Process detections
        for r in results.boxes:
            x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
            conf = r.conf[0].cpu().numpy()
            cls_id = int(r.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            # Person on motorcycle -> motorcycle
            if cls_name == "person":
                for i, l in enumerate(labels):
                    if l == "motorcycle":
                        boxes[i] = [x1,y1,x2,y2]
                        cls_name = "motorcycle"
                        break

            if is_vehicle_class(cls_name):
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_name)

                cx, cy = centroid([x1, y1, x2, y2])
                if cx > entry_line and cx < exit_line:
                    vehicles_in_region += 1

            if is_violation(cls_name, [x1, y1, x2, y2]):
                violation_flag = True

            # Draw bounding box
            color = (0,255,0) if cls_name!="person" else (0,0,255)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1),int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Collision detection within this frame
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if check_collision(boxes[i], boxes[j]):
                    collision_flag = True
                    cv2.line(frame, centroid(boxes[i]), centroid(boxes[j]), (0,0,255), 2)

        total_vehicles += vehicles_in_region
        frames.append(cv2.resize(frame, (640,480)))

    # Combine 4 frames into 2x2 grid
    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    grid = np.vstack([top_row, bottom_row])

    # -----------------------
    # Overlay information
    # -----------------------
    cv2.putText(grid, f"Total Vehicles: {total_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(grid, f"Collision: {'Yes' if collision_flag else 'No'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if collision_flag else (0,255,0), 2)
    cv2.putText(grid, f"Violation: {'Yes' if violation_flag else 'No'}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if violation_flag else (0,255,0), 2)

    # Show live annotated video
    cv2.imshow("4 Camera Feeds", grid)

    # Publish MQTT output
    output_data = {
        "vehicles_in_region": total_vehicles,
        "collision": "yes" if collision_flag else "no",
        "violations": "yes" if violation_flag else "no"
    }
    client.publish(TOPIC, json.dumps(output_data))
    print("Published:", output_data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
