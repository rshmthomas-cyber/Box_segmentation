from ultralytics import YOLO
import cv2

# 1. Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  

# 2. Start video capture (0 is default webcam)
cap = cv2.VideoCapture(0)

# Set resolution to wide (e.g., 1280x720 or 1920x1080 if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO inference
    # For very distorted lenses, consider preprocessing to undistort
    results = model(frame)

    # 4. Visualize the results
    annotated_frame = results[0].plot()
    cv2.imshow("Wide Angle YOLO Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
