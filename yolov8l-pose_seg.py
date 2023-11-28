import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path
import torch

# def overlay_annotated_crop(img_annotated, pose_annotated, x_min, y_min, x_max, y_max):
#     """Overlay the annotated pose crop onto the original image."""
    
    

def pose_estimation(model, source, view_img=False, save_img=False, exist_ok=False):
    pose_model = YOLO('models/yolov8l-pose.pt')
    seg_model = YOLO('models/yolov8l-seg.pt')
    image = cv2.imread(source)

    if not Path(source).exists():  # Check source path
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    seg_results = seg_model.track(image,
                        conf=0.4, 
                        iou=0.7, 
                        classes=[0],
                        device='cpu',
                        # save=True, 
                        # visualize=True,
                        )
    bboxes = seg_results[0].boxes
    segmentation_masks = seg_results[0].masks

    if segmentation_masks is not None:
        img_annotated = image.copy()  # Make a copy for annotations
        for i, bbox in enumerate(bboxes):
            bbox_coords = bbox.xyxy.numpy()
            if bbox_coords.ndim == 2 and bbox_coords.shape[0] == 1:
                bbox_coords = bbox_coords.flatten()

            if bbox_coords.size == 4:
                x_min, y_min, x_max, y_max = map(int, bbox_coords)
                cropped_image = image[y_min:y_max, x_min:x_max]
                pose_results = pose_model(cropped_image)
                print("Type of pose_results:", type(pose_results))
                print("Pose results content:", pose_results)
                print("Pose results plot method:", pose_results[0].plot)
                # Annotate the pose results on the img_annotated
                if hasattr(pose_results, 'plot'):
                    # Assuming pose_results.plot returns an annotated image
                    pose_annotated = pose_results.plot(boxes=True)
                    # Here you need to overlay pose_annotated back onto img_annotated
                    # This requires a method to correctly position the annotated crop back onto the original image
                    # img_annotated = overlay_annotated_crop(img_annotated, pose_annotated, x_min, y_min, x_max, y_max)
                else:
                    print("The returned pose result does not have a 'plot' method.")
            else:
                print(f"Unexpected format for bbox coordinates: {bbox_coords}")

    else:
        img_annotated = image
        print("No segmentation masks found. Using original pose detection results.")
    

    if view_img:
        cv2.imshow("Image Pose Estimation", img_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    
    if save_img:
        save_dir = Path('output_seg_pose') / 'exp'
        save_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(source).stem + '_pose'
        img_filename = Path(save_dir / base_filename).with_suffix('.jpg')
        unique_filename = increment_path(img_filename, exist_ok)
        cv2.imwrite(str(unique_filename), img_annotated)
        print(f"Saving image: {unique_filename}")  # Debug print

def is_video_file(filepath):
    """Check if the file is a video based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(filepath.endswith(ext) for ext in video_extensions)

 
def process_folder(model_path, folder_path, save_img=False, exist_ok=False):
    model = YOLO(model_path)  # Create the model object once here
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            pose_estimation(model, file_path, view_img=False, save_img=save_img, exist_ok=exist_ok)

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8l-pose.pt', help='initial weights path')
    parser.add_argument('--source', type=str,default='People-Walking-2.mp4' , help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


def main(opt):
    if os.path.isdir(opt.source):
        process_folder(opt.model, opt.source, save_img=opt.save_img, exist_ok=opt.exist_ok)
    elif os.path.isfile(opt.source):
        if isinstance(opt.model, str):
            model = YOLO(opt.model)  # Instantiate the model for a single file
        else:
            model = opt.model
        pose_estimation(model, opt.source, view_img=opt.view_img, save_img=opt.save_img, exist_ok=opt.exist_ok)
    else:
        raise FileNotFoundError(f"Source '{opt.source}' does not exist as a file or directory.")


    # pose_estimation(**vars(opt))

if __name__ == '__main__':

    opt = parse_opt()

    main(opt)

