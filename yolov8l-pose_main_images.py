import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path

# def preprocess_image(image, model_img_size=640):
#     # Calculate the ratio of the target dimensions to the original dimensions
#     h, w = image.shape[:2]
#     scale = model_img_size / max(h, w)
    
#     # Scale the image dimensions to the model's expected size
#     h_scaled, w_scaled = int(h * scale), int(w * scale)
#     image_resized = cv2.resize(image, (w_scaled, h_scaled))

#     # Calculate padding to maintain aspect ratio
#     top_pad = (model_img_size - h_scaled) // 2
#     bottom_pad = model_img_size - h_scaled - top_pad
#     left_pad = (model_img_size - w_scaled) // 2
#     right_pad = model_img_size - w_scaled - left_pad

#     # Apply padding
#     image_padded = cv2.copyMakeBorder(image_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     return image_padded


def preprocess_image(image, model_img_size=640):
    """
    Sharpen the image to enhance edges without applying edge detection filters.
    This function uses a kernel to apply a sharpening filter to the image and resizes it to fit the model input size.
    
    :param image: Input image in BGR format
    :param model_img_size: Size (width, height) to resize the image to, should match the model input size
    :return: Sharpened and resized image
    """
    # Define the sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    # Resize image to fit model input size
    image_resized = cv2.resize(sharpened_image, (model_img_size, model_img_size))

    return sharpened_image

# Replace the previous preprocess_image function with


# This function now focuses on edge enhancement which is helpful for the model to detect features.
# Replace the previous preprocess_image function with this one and feed the output to the model.


def pose_estimation(model, source, view_img=False, save_img=False, exist_ok=False):
    
    if isinstance(model, str):  
        model = YOLO(model)  
    
    if not Path(source).exists(): # Check source path
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    
    # if is_video_file(source):
    # else:
    image = cv2.imread(source)
    image_processed = preprocess_image(image, model_img_size=640)
    # for False positives
    results = model.track(image_processed, 
                    # save=True, 
                    # conf=0.4, 
                    # iou=0.5, 
                    device='cpu',
                    # augment=True,
                    # retina_masks=True,
                    # visualize=True,
                    )

    # results = model.track(image_processed, 
    #         # save=True, 
    #         conf=0.4, 
    #         iou=0.7, 
    #         device='cpu',
    #         augment=True,
    #         # retina_masks=True,
    #         # persist=True,
    #         # visualize=True,
    #         )
    img_annotated = results[0].plot(boxes=True)

    if view_img:
        cv2.imshow("Image Pose Estimation", img_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        save_dir = Path('output_pose') / 'exp'
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
    # parse if video
    parser.add_argument('--video', action='store_true', help='video file path')
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

