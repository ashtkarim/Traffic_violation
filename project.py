import cv2
import numpy as np
from ultralytics import YOLO

def is_red_light(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return False
    red_ratio = red_pixels / total_pixels
    return red_ratio > 0.1

# Load models
model_crosswalk = YOLO("crosswalk.pt")  # for danger line detection (first frame)
model_yolo = YOLO("yolov8n.pt")         # for cars + traffic lights detection

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Detect danger line from first frame
ret, first_frame = cap.read()
danger_line_y = None
if ret:
    results = model_crosswalk(first_frame, conf=0.3)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        top_y_coords = [box.xyxy[0][1].item() for box in boxes]
        danger_line_y = int(min(top_y_coords))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw danger line
    if danger_line_y is not None:
        cv2.line(frame, (0, danger_line_y), (frame.shape[1], danger_line_y), (0, 0, 255), 2)
        cv2.putText(frame, 'Danger Line', (10, danger_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    results = model_yolo(frame, conf=0.3)
    boxes = results[0].boxes
    class_names = model_yolo.names

    # Filter traffic lights on left half above or near danger line
    traffic_lights = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        class_name = class_names[cls_id]
        if class_name.lower() == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_center_x = (x1 + x2) // 2
            # Condition: on left half and above or close to danger line
            if box_center_x > frame.shape[1] // 2 and (danger_line_y is None or y2 < danger_line_y + 20):
                traffic_lights.append((box, (x1, y1, x2, y2)))

    biggest_tl = None
    max_area = 0
    for box, (x1, y1, x2, y2) in traffic_lights:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            biggest_tl = (box, (x1, y1, x2, y2))

    red_light_on = False
    if biggest_tl is not None:
        box, (x1, y1, x2, y2) = biggest_tl
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0 and is_red_light(roi):
            red_light_on = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "Red Light", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Traffic Light", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw cars under danger line (optional, remove if not needed)
    for box in boxes:
        cls_id = int(box.cls[0].item())
        class_name = class_names[cls_id]
        if class_name.lower() == "car":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if danger_line_y is not None and y1 > danger_line_y:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Car", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Optional warning when red light detected
    if red_light_on:
        cv2.putText(frame, "STOP! Red Light ON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("Traffic Light and Danger Line Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
