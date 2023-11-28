from ultralytics import YOLO
import cv2

def pose_estimation_movie(input_file, output_file, model):

    cap = cv2.VideoCapture(input_file)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Tracking with default tracker
            results = model.track(frame, 
                                    save=True, 
                                    conf=0.25, 
                                    iou=0.7, 
                                    persist=True, 
                                    show=True)  

            img_annotated = results[0].plot(boxes=True)

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
