import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("crosswalk.pt")

# Load video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read the first frame to get danger line position
ret, first_frame = cap.read()
danger_line_y = None

if ret:
    results = model(first_frame, conf=0.30)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        top_y_coords = [box.xyxy[0][1].item() for box in boxes]
        danger_line_y = int(min(top_y_coords))

# Rewind video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process full video, but just draw the stored danger line
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the danger line from the first frame
    if danger_line_y is not None:
        cv2.line(frame, (0, danger_line_y), (frame.shape[1], danger_line_y), (0, 0, 255), 2)
        cv2.putText(frame, 'Danger Line', (10, danger_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Write and show
    out.write(frame)
    cv2.imshow("Video with Danger Line", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
