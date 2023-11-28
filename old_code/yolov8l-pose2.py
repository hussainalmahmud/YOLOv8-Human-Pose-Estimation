from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8x-pose.pt')  # Load an official Pose model

source = 'People-Walking-2.mp4'

# Perform tracking with the model
results = model.track(source, save=True, conf=0.3, iou=0.7, persist=True, show=True)  # Tracking with default tracker