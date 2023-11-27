import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path



def pose_estimation(model, source, view_img=False, save_img=False, exist_ok=False):
    
    if isinstance(model, str):  
        model = YOLO(model)  
    
    if not Path(source).exists(): # Check source path
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    
    if is_video_file(source):
        # Video setup
        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # Output setup
        save_dir = increment_path(Path('output') / 'exp', exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(save_dir / f'{Path(source).stem}.mp4')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Initialize tracking history
        track_history = {}

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Object detection and tracking
                results = model.track(frame, 
                                    # save=True, 
                                    conf=0.1, 
                                    iou=0.7, 
                                    persist=True,
                                    device='mps',
                                    )

                img_annotated = results[0].plot(boxes=True)

                # Process tracking lines
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 

                        # Get or create track history for this track_id
                        track = track_history.get(track_id, [])
                        track.append((float(bbox_center[0]), float(bbox_center[1])))

                        if len(track) > 30:
                            track.pop(0)

                        track_history[track_id] = track

                        # Draw tracking line
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_annotated, [points], isClosed=False, color=(0, 0, 255), thickness=2)


                if view_img:
                    cv2.imshow("Video Pose Estimation", img_annotated)

                if save_img:
                    video_writer.write(img_annotated)
                    

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Image processing
        image = cv2.imread(source)
        img_annotated = process_image(model, image)

        if view_img:
            cv2.imshow("Image Pose Estimation", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_img:
            # original
            # save_dir = increment_path(Path('output') / 'exp', exist_ok)
            # save_dir.mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(str(save_dir / f'{Path(source).stem}_pose.jpg'), img_annotated)
            save_dir = Path('output') / 'exp'
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

def process_image(model, image):
    """Process a single image for pose estimation."""
    results = model.track(image, 
                        save=True, 
                        conf=0.1, 
                        iou=0.5, 
                        device='cpu')

    img_annotated = results[0].plot(boxes=True)
    return img_annotated
 
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

