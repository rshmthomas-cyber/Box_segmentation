from ultralytics import YOLO
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load your trained segmentation model
model = YOLO(r"D:\Raspberry Pi\best (1).pt")

# Path to test image
image_path = r"D:\Raspberry Pi\BOX_SEGMENTATION_DATASET_TEMPLATE\Labeled_Project\YOLOv8_Format\basket2.v1i.yolov8\test\images\000200_jpg.rf.65589a6ff5499c3ca30191072c72d710.jpg"
img = cv2.imread(image_path)

# Run prediction
results = model.predict(source=img, conf=0.3, verbose=False)
r = results[0]

# If no detections, skip gracefully
if r.masks is None or r.boxes is None or len(r.boxes) == 0:
    print("⚠️ No objects detected in this image.")
else:
    for box, mask, conf, cls in zip(r.boxes.xyxy, r.masks.data, r.boxes.conf, r.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        conf = float(conf)
        cls_name = model.names[int(cls)]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Apply segmentation mask
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

    from google.colab.patches import cv2_imshow
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
