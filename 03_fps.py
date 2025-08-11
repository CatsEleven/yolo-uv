import cv2
import numpy as np
from ultralytics import YOLO, SAM
from pathlib import Path
import shutil
import glob

# --- Configuration ---
SOURCE_VIDEO_PATH = Path("movie/source/zi.mp4")
TEMP_FRAME_DIR = Path("temp1")
PROCESSED_FRAME_DIR = Path("temp2")
DEST_DIR = Path("movie/dest")
OUTPUT_VIDEO_PATH = DEST_DIR / f"{SOURCE_VIDEO_PATH.stem}_cartoon.mp4"
TARGET_FPS = 10

# --- Setup Directories ---
TEMP_FRAME_DIR.mkdir(exist_ok=True)
PROCESSED_FRAME_DIR.mkdir(exist_ok=True)
DEST_DIR.mkdir(exist_ok=True)

# --- Model Loading ---
print("Loading models...")
yolo_model = YOLO("model/yolov8n.pt")
sam_model = SAM("model/sam2.1_b.pt")
print("Models loaded.")

def process_image_to_cartoon(image_bgr, yolo_model, sam_model):
    """
    Applies a cartoon effect to a single image frame.
    Returns a tuple (processed_image, success_flag).
    """
    # YOLO Object Detection
    yolo_results = yolo_model.predict(image_bgr, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    clses = yolo_results[0].boxes.cls.cpu().numpy()
    person_boxes = [box.astype(int) for box, c in zip(boxes, clses) if int(c) == 0]

    result_image = image_bgr.copy()

    def posterize(img, k=8):
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        new_img = centers[labels.flatten()]
        return new_img.reshape(img.shape)

    if not person_boxes:
        return None, False # No person detected

    processed_at_least_one = False
    for bbox in person_boxes:
        # SAM Segmentation
        sam_results = sam_model.predict(image_bgr, bboxes=bbox, verbose=False)
        if not sam_results or not sam_results[0].masks:
            continue
        
        processed_at_least_one = True
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        person = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        cartoon_color = posterize(person, k=8)
        blurred = cv2.GaussianBlur(cartoon_color, (51, 51), 0)
        gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = blurred.copy()
        cartoon[edges[:, :, 0] > 100] = 0
        
        bg = cv2.bitwise_and(result_image, result_image, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(cartoon, cartoon, mask=mask)
        result_image = cv2.add(bg, fg)
        
    if processed_at_least_one:
        return result_image, True
    else:
        return None, False

# --- Step 1: Video to Frames ---
print(f"Step 1: Extracting frames from {SOURCE_VIDEO_PATH}...")
cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / TARGET_FPS)
frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        output_frame_path = TEMP_FRAME_DIR / f"frame_{saved_frame_count:06d}.png"
        cv2.imwrite(str(output_frame_path), frame)
        saved_frame_count += 1
    frame_count += 1

cap.release()
print(f"Extracted {saved_frame_count} frames to {TEMP_FRAME_DIR}.")

# --- Step 2: Process Frames ---
print(f"Step 2: Processing frames in {TEMP_FRAME_DIR}...")
frame_files = sorted(glob.glob(str(TEMP_FRAME_DIR / "*.png")))

for i, frame_path in enumerate(frame_files):
    print(f"Processing frame {i+1}/{len(frame_files)}: {frame_path}")
    image = cv2.imread(frame_path)
    processed_image, success = process_image_to_cartoon(image, yolo_model, sam_model)
    if success:
        output_processed_path = PROCESSED_FRAME_DIR / Path(frame_path).name
        cv2.imwrite(str(output_processed_path), processed_image)
    else:
        print(f"  -> Detection failed. Skipping frame.")

print(f"Processed {len(frame_files)} frames and saved to {PROCESSED_FRAME_DIR}.")

# --- Step 3: Frames to Video ---
print(f"Step 3: Combining processed frames into video...")
processed_frame_files = sorted(glob.glob(str(PROCESSED_FRAME_DIR / "*.png")))

if processed_frame_files:
    first_frame = cv2.imread(processed_frame_files[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, TARGET_FPS, (width, height))

    for frame_path in processed_frame_files:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {OUTPUT_VIDEO_PATH}.")
else:
    print("No processed frames to create a video.")

# --- Step 4: Cleanup ---
print("Cleaning up temporary files...")
for dir_to_clean in [TEMP_FRAME_DIR, PROCESSED_FRAME_DIR]:
    files = glob.glob(str(dir_to_clean / '*'))
    for f in files:
        try:
            if Path(f).is_file():
                Path(f).unlink()
        except Exception as e:
            print(f"Error while deleting file {f}: {e}")
print("Temporary files removed.")

print("Processing complete.")
