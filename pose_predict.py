import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path


def pose_estimation(
    model, source, is_video=False, view_img=False, save_img=False, exist_ok=False
):
    if source != 0 and not Path(str(source)).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if is_video or source == 0:
        # Video setup
        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        # Output setup
        save_dir = increment_path(Path("output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_filename = (
            "webcam_output.mp4" if source == 0 else f"{Path(source).stem}.mp4"
        )
        output_path = str(save_dir / output_filename)
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

        track_history = {}
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = model.track(
                    frame,
                    conf=0.5,
                    iou=0.7,
                    device="cpu",
                    imgsz=640,
                    tracker="bytetrack.yaml",
                    persist=True,  # set persist to True for tracking in video (Re-Identification)
                    retina_masks=True,
                    augment=True,
                )

                img_annotated = results[0].plot(boxes=True)

                # Process tracking lines
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        if source != 0:
                            # Get or create track history for this track_id
                            track = track_history.get(track_id, [])
                            track.append((float(bbox_center[0]), float(bbox_center[1])))

                            if len(track) > 10:
                                track.pop(0)

                            track_history[track_id] = track

                            # Draw tracking line
                            points = (
                                np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            )
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
    else:  # if image file
        image = cv2.imread(source)

        results = model.predict(
            image,
            conf=0.5,
            iou=0.7,
            device="cpu",
            imgsz=640,
            tracker="bytetrack.yaml",
            retina_masks=True,
            augment=True,
        )

        img_annotated = results[0].plot(boxes=True)

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


def process_folder(model_path, folder_path, save_img=False, exist_ok=False):
    model = YOLO(model_path)  # Create the model object once here
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            pose_estimation(
                model, file_path, view_img=False, save_img=save_img, exist_ok=exist_ok
            )


def parse_opt():
    """
    Parse command line arguments.

    Example use:
    python pose_predict.py --model yolov8l-pose.pt --source 0 --is_video --view-img # webcam
    python pose_predict.py --model yolov8l-pose.pt --source video.mp4 --is_video --save-img --view-img
    python pose_predict.py --model yolov8l-pose.pt --source img_name.jpg --save-img --view-img
    python pose_predict.py --model yolov8l-pose.pt --source folder_name --save-img

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="yolov8l-pose.pt", help="initial weights path"
    )
    parser.add_argument(
        "--source", type=str, default="People-Walking-2.mp4", help="video file path"
    )
    parser.add_argument(
        "--is_video", action="store_true", help="specify if file is video"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    return parser.parse_args()


def main(local_opt):
    if os.path.isdir(local_opt.source):  # if folder
        process_folder(
            local_opt.model,
            local_opt.source,
            save_img=local_opt.save_img,
            exist_ok=local_opt.exist_ok,
        )
    elif local_opt.source == "0":  # if webcam
        model = YOLO(local_opt.model)
        pose_estimation(
            model,
            int(local_opt.source),
            is_video=local_opt.is_video,
            view_img=local_opt.view_img,
            save_img=local_opt.save_img,
            exist_ok=local_opt.exist_ok,
        )
    elif os.path.isfile(local_opt.source):  # if file
        model = (
            YOLO(local_opt.model)
            if isinstance(local_opt.model, str)
            else local_opt.model
        )
        pose_estimation(
            model,
            local_opt.source,
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
