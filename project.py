import cv2
import numpy as np
import math
from ultralytics import YOLO

# ─── Centroid Tracker ───
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.max_distance = max_distance
        self.violations = set()

    def update(self, boxes):
        updated_ids = {}
        new_centroids = []
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            new_centroids.append((cx, cy, x1, y1, x2, y2))

        for (cx, cy, x1, y1, x2, y2) in new_centroids:
            assigned = False
            for obj_id, (ox, oy) in self.objects.items():
                if math.hypot(cx - ox, cy - oy) < self.max_distance:
                    self.objects[obj_id] = (cx, cy)
                    updated_ids[(x1, y1, x2, y2)] = obj_id
                    assigned = True
                    break
            if not assigned:
                self.objects[self.next_id] = (cx, cy)
                updated_ids[(x1, y1, x2, y2)] = self.next_id
                self.next_id += 1
        return updated_ids

# ─── Detect Color of Traffic Light ───
def detect_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return 'unknown'

    red_ratio = cv2.countNonZero(red_mask) / total_pixels
    yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
    green_ratio = cv2.countNonZero(green_mask) / total_pixels

    if red_ratio > 0.1:
        return 'red'
    elif yellow_ratio > 0.1:
        return 'yellow'
    elif green_ratio > 0.1:
        return 'green'
    else:
        return 'unknown'

# ─── Load YOLO Models ───
model_crosswalk = YOLO("crosswalk.pt")
model_yolo = YOLO("yolo12n.pt")

# ─── Load Video ───
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 20.0,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# ─── Detect Danger Line ───
ret, first_frame = cap.read()
danger_line_y = None
if ret:
    res = model_crosswalk(first_frame, conf=0.3)
    boxes = res[0].boxes
    if boxes:
        danger_line_y = int(min(b.xyxy[0][1].item() for b in boxes))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ─── Tracking Setup ───
frame_count = 0
last_detected_color = 'unknown'
tracker = CentroidTracker()

# ─── Main Loop ───
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    H, W = frame.shape[:2]

    # Draw danger line
    if danger_line_y is not None:
        cv2.line(frame, (0, danger_line_y), (W, danger_line_y), (0, 0, 255), 2)
        cv2.putText(frame, "Danger Line", (10, danger_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # YOLO Detection
    res = model_yolo(frame, conf=0.1)
    boxes = res[0].boxes
    names = model_yolo.names

    # ─── Traffic Light Detection ───
    biggest_box = None
    max_area = 0
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if names[cls_id].lower() == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x1 + x2) // 2 < W // 2:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                biggest_box = (x1, y1, x2, y2)

    red_light_on = False
    if biggest_box is not None:
        x1, y1, x2, y2 = biggest_box
        roi = frame[y1:y2, x1:x2]
        if frame_count % 5 == 0:
            last_detected_color = detect_light_color(roi)
        color_name = last_detected_color

        if color_name == 'red':
            red_light_on = True
            box_color = (0, 0, 255)
            label = "Red Light"
        elif color_name == 'yellow':
            box_color = (0, 255, 255)
            label = "Yellow Light"
        elif color_name == 'green':
            box_color = (0, 255, 0)
            label = "Green Light"
        else:
            box_color = (128, 128, 128)
            label = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    # ─── Car Detection Below Danger Line ───
    car_boxes = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if names[cls_id].lower() == "car":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if danger_line_y is None or y2 > danger_line_y:  # Only cars below the line
                car_boxes.append((x1, y1, x2, y2))

    tracked_ids = tracker.update(car_boxes)

    for (x1, y1, x2, y2) in car_boxes:
        car_id = tracked_ids.get((x1, y1, x2, y2), None)
        if car_id is None:
            continue

        # Check for Red Light Violation
        violation = False
        if red_light_on and danger_line_y is not None and y1 <= danger_line_y <= y2:
            tracker.violations.add(car_id)
            violation = True

        color = (0, 0, 255) if violation else (0, 255, 0)
        label = f"Car {car_id}" + (" - VIOLATION" if violation else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ─── Output Frame ───
    out.write(frame)
    cv2.imshow("Traffic Light Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── End and Report ───
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nRed Light Violations:")
for car_id in tracker.violations:
    print(f"Car {car_id} violated the red light.")
