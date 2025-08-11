from ultralytics import YOLO
from pathlib import Path

model = YOLO("model/yolov8n.pt")

source_file_name = "side.png"
source_path = Path("yolo/source") / source_file_name

results = model(source_path)

output_path = Path("yolo/dest") / "test-deteted.jpg"

results[0].save(filename=output_path)

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}, BBox(xyxy): {xyxy}")

print(f"Detection result saved to: {output_path}")
