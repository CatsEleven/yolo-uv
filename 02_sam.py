import cv2
import numpy as np
from ultralytics import YOLO, SAM
from pathlib import Path

# --- Path Settings ---
SOURCE_DIR = Path("yolo/source")
DEST_DIR = Path("yolo/dest")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# --- Input ---
# このファイル名を変更して、処理する画像を指定します
source_file_name = "fast.png"
source_path = SOURCE_DIR / source_file_name
output_path = DEST_DIR / f"{source_path.stem}_detected.png"

# --- Model Loading ---
yolo_model = YOLO("model/yolov8n.pt")
sam_model = SAM("model/sam2.1_b.pt")

# --- Image Loading ---
image_bgr = cv2.imread(str(source_path))
if image_bgr is None:
    print(f"Error: Could not read image from {source_path}")
    exit()

# --- YOLO Object Detection ---
yolo_results = yolo_model(source_path)
boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
clses = yolo_results[0].boxes.cls.cpu().numpy()

# Filter for 'person' class (class id 0)
person_boxes = [box.astype(int) for box, c in zip(boxes, clses) if int(c) == 0]

# --- Image Processing ---
result = image_bgr.copy()

def posterize(img, k=8):
    """Applies posterization to an image."""
    data = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    new_img = centers[labels.flatten()]
    return new_img.reshape(img.shape)

# Process each detected person
for idx, bbox in enumerate(person_boxes):
    sam_results = sam_model(source_path, bboxes=bbox)
    mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    person = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    
    cartoon_color = posterize(person, k=8)
    blurred = cv2.GaussianBlur(cartoon_color, (51, 51), 0)
    
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    cartoon = blurred.copy()
    cartoon[edges[:,:,0]>100] = 0
    
    mask3 = cv2.merge([mask, mask, mask])
    bg = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(cartoon, cartoon, mask=mask)
    result = cv2.add(bg, fg)

# --- Save Result ---
cv2.imwrite(str(output_path), result)
print(f"Image saved to {output_path}")
