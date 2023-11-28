import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def visualize_keypoints(image, keypoints):
    for kp in keypoints:
        # Example: assuming kp has attributes 'x' and 'y'
        # You'll need to adjust this based on the actual structure of kp
        x, y = kp.x, kp.y
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw green circle

def pose_estimation(model, source, view_img=False, save_img=False, exist_ok=False):
    pose_model = YOLO('models/yolov8l-pose.pt')
    seg_model = YOLO('models/yolov8l-seg.pt')

    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    image = cv2.imread(source)
    seg_results = seg_model.track(image, conf=0.4, iou=0.7, classes=[0], device='cpu')
    bboxes = seg_results[0].boxes
    segmentation_masks = seg_results[0].masks

    img_annotated = image.copy()

    if segmentation_masks is not None:
        for bbox in bboxes:
            bbox_coords = bbox.xyxy.numpy()
            if bbox_coords.ndim == 2 and bbox_coords.shape[0] == 1:
                bbox_coords = bbox_coords.flatten()

            if bbox_coords.size == 4:
                x_min, y_min, x_max, y_max = map(int, bbox_coords)
                cropped_image = image[y_min:y_max, x_min:x_max]
                pose_results = pose_model(cropped_image)

                # Check if pose_results has keypoints and visualize them
                if hasattr(pose_results, 'keypoints'):
                    visualize_keypoints(img_annotated, pose_results.keypoints)
                else:
                    print("No keypoints found in pose results.")
            else:
                print("Unexpected format for bbox coordinates:", bbox_coords)
    else:
        print("No segmentation masks found. Using original pose detection results.")

    # Save or show image
    if save_img:
        cv2.imwrite("annotated_image.jpg", img_annotated)

    if view_img:
        cv2.imshow("Pose Estimation", img_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
pose_estimation(None, source='/Users/hussain/github_repo/YOLOv8-Human-Pose-Estimation/4142.jpg', view_img=True, save_img=True)
