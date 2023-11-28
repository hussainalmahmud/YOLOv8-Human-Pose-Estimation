from ultralytics import YOLO
import cv2
import dlib

def pose_estimation_movie(input_file, output_file, model):
    cap = cv2.VideoCapture(input_file)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    # Initialize object trackers
    trackers = []

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Object detection
            results = model.track(frame, 
                                save=True, 
                                conf=0.1, 
                                iou=0.7, 
                                persist=True, 
                                show=True,
                                tracker="bytetrack.yaml")  

            img_annotated = results[0].plot(boxes=True)

            # Initialize or update trackers for detected objects
            for i, det in enumerate(results[0].boxes.xyxy.cpu()):
                bbox = tuple(map(int, det.tolist()))
                if len(trackers) <= i:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(*bbox))
                    trackers.append(tracker)
                else:
                    tracker = trackers[i]
                    tracker.update(frame)
                    track_bbox = tracker.get_position()
                    bbox = (int(track_bbox.left()), int(track_bbox.top()), int(track_bbox.right()), int(track_bbox.bottom()))

                cv2.rectangle(img_annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.imshow("Video Pose Estimation", img_annotated)

            video.write(img_annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    input_file = 'People-Walking-2.mp4'
    output_file = 'out-movie.mp4'

    model = YOLO('yolov8l-pose.pt')

    pose_estimation_movie(input_file, output_file, model)
