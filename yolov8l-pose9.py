import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path

def pose_estimation(model, source, view_img=False, save_img=False, exist_ok=False):
    
    pose_model = YOLO(model)
    det_model = YOLO('models/yolov8l.pt')
    
    
    if not Path(source).exists(): # Check sourcwqe path
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
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
            det_results = det_model.track(frame, save=True, conf=0.25, iou=0.7, persist=True,device='cpu', classes=[0])
            pose_results = pose_model.track(frame, save=True, conf=0.25, iou=0.7, persist=True,device='cpu')
            # for det_box, pose_box in zip(det_results[0].boxes, pose_results[0].boxes):
            #     avg_conf = (det_box.conf + pose_box.conf) / 2
            #     # print("avg_conf ", avg_conf)
            img_annotated_det = det_results[0].plot()
            img_annotated_pose = pose_results[0].plot()
            alpha = 0.5  # Alpha controls the transparency: 0 is fully transparent, 1 is fully opaque
            img_annotated = cv2.addWeighted(img_annotated_det, alpha, img_annotated_pose, 1-alpha, 0)

                
            if det_results[0].boxes.id is not None:
                boxes = det_results[0].boxes.xyxy.cpu()
                track_ids = det_results[0].boxes.id.int().cpu().tolist()

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

    return



def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov8l-pose.pt', help='initial weights path')
    parser.add_argument('--source', type=str,default='People-Walking-2.mp4' , help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


def main(opt):
    """Main function."""
    pose_estimation(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

