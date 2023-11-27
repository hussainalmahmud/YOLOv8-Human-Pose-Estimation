import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
from sahi import AutoDetectionModel # SAHI: Slicing Aided Hyper Inference
from sahi.predict import get_sliced_prediction
# from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path

def run(weights='yolov8l-pose.pt', source='test.mp4', view_img=False, save_img=False, exist_ok=False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # yolov8_model_path = f'models/{weights}'
    # download_yolov8s_model(yolov8_model_path)
    # yolov8_model_path = YOLO(weights)
    yolov8_model_path = Path('models') / weights  # Example: 'models/yolov8l-pose.pt'
    model = YOLO('models/yolov8l-pose.pt')
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                        model_path=yolov8_model_path,
                                                        confidence_threshold=0.1
                                                        )

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = increment_path(Path('ultralytics_results_with_sahi') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))
    
    # Initialize tracking history
    track_history = {}
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = get_sliced_prediction(frame,
                                        detection_model,
                                        slice_height=512,
                                        slice_width=512,
                                        overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2)
        object_prediction_list = results.object_prediction_list

        results_pose = model.track(frame, 
                                save=True, 
                                conf=0.1, 
                                iou=0.7, 
                                persist=True, 
                                show=True)
        img_annotated = results_pose[0].plot(boxes=True)
        # Process tracking lines
        if results_pose[0].boxes.id is not None:
            boxes = results_pose[0].boxes.xyxy.cpu()
            track_ids = results_pose[0].boxes.id.int().cpu().tolist()

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

        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
                object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),-1)
            cv2.putText(frame,
                        label, (int(x1), int(y1) - 2),
                        0,
                        0.6, [255, 255, 255],
                        thickness=1,
                        lineType=cv2.LINE_AA)



        if view_img:
            # cv2.imshow(Path(source).stem, frame)
            # cv2.imshow(Path(source).stem, img_annotated)
            # cv2.imshow("Original - " + Path(source).stem, frame)
            # cv2.imshow("Annotated - " + Path(source).stem, img_annotated)
            combined_img = np.concatenate((frame, img_annotated), axis=1)
            # cv2.imshow("Original - " + Path(source).stem + " | Annotated - " + Path(source).stem, combined_img)
            cv2.imshow("Original and Annotated", combined_img)
            
        if save_img:
            # video_writer.write(frame)
            video_writer.write(img_annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8l-pose.pt', help='initial weights path')
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
