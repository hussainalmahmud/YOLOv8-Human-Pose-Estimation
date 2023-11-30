import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path
import os


def pose_estimation(
    pose_model,
    seg_model,
    source,
    is_video=False,
    view_img=False,
    save_img=False,
    exist_ok=False,
):
    # pose_model = YOLO('models/yolov8l-pose.pt')
    # seg_model = YOLO('models/yolov8l-seg.pt')

    if not Path(source).exists():  # Check sourcwqe path
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    if is_video:
        # Video setup
        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        # Output setup
        save_dir = increment_path(Path("output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(save_dir / f"{Path(source).stem}.mp4")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

        # Initialize tracking history
        track_history = {}

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Object detection and tracking
                det_results = seg_model.track(
                    frame,
                    save=True,
                    conf=0.25,
                    iou=0.7,
                    persist=True,
                    device="cpu",
                    classes=[0],
                )
                pose_results = pose_model.track(
                    frame, save=True, conf=0.25, iou=0.7, persist=True, device="cpu"
                )

                img_annotated_det = det_results[0].plot()
                img_annotated_pose = pose_results[0].plot()
                alpha = 0.5  # Alpha controls the transparency: 0 is fully transparent, 1 is fully opaque
                img_annotated = cv2.addWeighted(
                    img_annotated_det, alpha, img_annotated_pose, 1 - alpha, 0
                )

                if det_results[0].boxes.id is not None:
                    boxes = det_results[0].boxes.xyxy.cpu()
                    track_ids = det_results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                        # Get or create track history for this track_id
                        track = track_history.get(track_id, [])
                        track.append((float(bbox_center[0]), float(bbox_center[1])))

                        if len(track) > 20:
                            track.pop(0)

                        track_history[track_id] = track

                        # Draw tracking line
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            img_annotated,
                            [points],
                            isClosed=False,
                            color=(0, 0, 255),
                            thickness=2,
                        )

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
        image = cv2.imread(source)
        det_results = seg_model.predict(
            image,
            conf=0.25,
            iou=0.7,
            device="cpu",
            classes=[0],
        )
        pose_results = pose_model.predict(
            image,
            conf=0.25,
            iou=0.7,
            #   save=True,
            device="cpu",
        )
        img_annotated_det = det_results[0].plot()
        img_annotated_pose = pose_results[0].plot()
        alpha = 0.5  # Alpha controls the transparency: 0 is fully transparent, 1 is fully opaque
        img_annotated = cv2.addWeighted(
            img_annotated_det, alpha, img_annotated_pose, 1 - alpha, 0
        )

        if view_img:
            cv2.imshow("Image Pose Estimation", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_img:
            save_dir = Path("output_pose") / "exp"
            save_dir.mkdir(parents=True, exist_ok=True)
            base_filename = Path(source).stem + "_pose"
            img_filename = Path(save_dir / base_filename).with_suffix(".jpg")
            unique_filename = increment_path(img_filename, exist_ok)
            cv2.imwrite(str(unique_filename), img_annotated)
            print(f"Saving image: {unique_filename}")  # Debug print


def is_video_file(filepath):
    """Check if the file is a video based on its extension."""
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    return any(filepath.endswith(ext) for ext in video_extensions)


def process_folder(
    pose_model,
    seg_model,
    folder_path,
    video=False,
    view_img=False,
    save_img=False,
    exist_ok=False,
):
    pose_model = YOLO(pose_model)
    seg_model = YOLO(seg_model)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            pose_estimation(
                pose_model=pose_model,
                seg_model=seg_model,
                source=file_path,
                is_video=video,
                view_img=view_img,
                save_img=save_img,
                exist_ok=exist_ok,
            )


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose_model",
        type=str,
        default="yolov8l-pose.pt",
        help="initial weights path for pose model",
    )
    parser.add_argument(
        "--seg_model",
        type=str,
        default="yolov8l-seg.pt",
        help="initial weights path for seg model",
    )
    parser.add_argument(
        "--source", type=str, default="People-Walking-2.mp4", help="video file path"
    )
    parser.add_argument("--is_video", action="store_true", help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    return parser.parse_args()


def main(local_opt):
    if os.path.isdir(local_opt.source):
        process_folder(
            local_opt.pose_model,
            local_opt.seg_model,
            local_opt.source,
            save_img=local_opt.save_img,
            exist_ok=local_opt.exist_ok,
        )
    elif os.path.isfile(local_opt.source):
        pose_model = YOLO(local_opt.pose_model)  # Instantiate the model for a single file
        seg_model = YOLO(local_opt.seg_model)
        pose_estimation(
            pose_model,
            seg_model,
            source=local_opt.source,
            is_video=local_opt.is_video,
            view_img=local_opt.view_img,
            save_img=local_opt.save_img,
            exist_ok=local_opt.exist_ok,
        )

    else:
        raise FileNotFoundError(
            f"Source '{local_opt.source}' does not exist as a file or directory."
        )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
