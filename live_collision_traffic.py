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
# Entry/Exit lines (y-coordinates) per camera
# -----------------------
entry_lines = [150, 150, 150, 150]  # adjust per camera
exit_lines  = [400, 400, 400, 400]

# -----------------------
# Centroid tracker per camera
# -----------------------
class CentroidTracker:
    def _init_(self, max_distance=50):
        self.objects = {}  # objectID -> centroid
        self.next_object_id = 0
        self.max_distance = max_distance
        self.count = 0

    def update(self, detections, entry_line, exit_line):
        updated_objects = {}
        for det in detections:
            x1, y1, x2, y2 = det
            c = (int((x1+x2)/2), int((y1+y2)/2))

            matched_id = None
            for obj_id, prev_c in self.objects.items():
                distance = np.linalg.norm(np.array(c)-np.array(prev_c))
                if distance < self.max_distance:
                    matched_id = obj_id
                    break

            if matched_id is None:
                # New vehicle enters
                if c[1] >= entry_line and c[1] <= exit_line:
                    self.count += 1
                self.objects[self.next_object_id] = c
                self.next_object_id += 1
            else:
                updated_objects[matched_id] = c
                # Vehicle leaves
                if c[1] > exit_line:
                    self.count = max(0, self.count-1)

        self.objects = updated_objects
        return self.count

trackers = [CentroidTracker(max_distance=50) for _ in range(4)]

# -----------------------
# Helper functions
# -----------------------
def check_collision(box1, box2):
    c1 = (int((box1[0]+box1[2])/2), int((box1[1]+box1[3])/2))
    c2 = (int((box2[0]+box2[2])/2), int((box2[1]+box2[3])/2))
    distance = np.linalg.norm(np.array(c1)-np.array(c2))
    return distance < 30  # tight threshold

def is_vehicle_class(cls_name):
    return cls_name in ["car", "truck", "bus", "motorcycle"]

def is_violation(cls_name, box):
    # Placeholder for basic violation detection
    return False

# -----------------------
# Main loop
# -----------------------
while True:
    frames = []
    per_camera_counts = []
    per_camera_collision = []
    per_camera_violation = []

    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480,640,3), dtype=np.uint8)

        results = model.predict(source=frame, imgsz=640, conf=0.3)[0]

        boxes = []
        labels = []
        current_detections = []

        # Draw entry/exit lines clearly
        cv2.line(frame, (0, entry_lines[idx]), (frame.shape[1], entry_lines[idx]), (255,255,0), 2)
        cv2.line(frame, (0, exit_lines[idx]), (frame.shape[1], exit_lines[idx]), (0,0,255), 2)

        collision_flag = False
        violation_flag = False

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
                boxes.append([x1,y1,x2,y2])
                labels.append(cls_name)
                current_detections.append([x1,y1,x2,y2])

            if is_violation(cls_name, [x1,y1,x2,y2]):
                violation_flag = True

            color = (0,255,0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Collision detection within this camera
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if check_collision(boxes[i], boxes[j]):
                    collision_flag = True
                    pt1 = (int((boxes[i][0]+boxes[i][2])/2), int((boxes[i][1]+boxes[i][3])/2))
                    pt2 = (int((boxes[j][0]+boxes[j][2])/2), int((boxes[j][1]+boxes[j][3])/2))
                    cv2.line(frame, pt1, pt2, (0,0,255), 2)

        # Update tracker for this camera
        cam_count = trackers[idx].update(current_detections, entry_lines[idx], exit_lines[idx])
        per_camera_counts.append(cam_count)
        per_camera_collision.append("Yes" if collision_flag else "No")
        per_camera_violation.append("Yes" if violation_flag else "No")

        # Overlay small text (no overlap)
        cv2.putText(frame, f"Count: {cam_count}", (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"Collision: {per_camera_collision[-1]}", (5,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) if collision_flag else (0,255,0), 1)
        cv2.putText(frame, f"Violation: {per_camera_violation[-1]}", (5,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) if violation_flag else (0,255,0), 1)

        frames.append(cv2.resize(frame, (640,480)))

    # Combine frames into 2x2 grid
    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    grid = np.vstack([top_row, bottom_row])

    cv2.imshow("4 Camera Feeds", grid)

    # Publish MQTT output
    output_data = {
        "per_camera_counts": per_camera_counts,
        "per_camera_collision": per_camera_collision,
        "per_camera_violation": per_camera_violation
    }
    client.publish(TOPIC, json.dumps(output_data))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
