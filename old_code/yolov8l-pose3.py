from ultralytics import YOLO
import cv2

def pose_estimation_movie(filename_in, filename_out, model):
    # Start capturing video
    cap = cv2.VideoCapture(filename_in)

    # Video file save settings
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(filename_out, fourcc, fps, (w, h))

    # Threshold for ground detection
    ground_threshold = h * 0.75  

    while cap.isOpened():
        # Extract frame
        success, frame = cap.read()

        if success:
            # Object detection
            results = model.track(frame, 
                            save=True, 
                            conf=0.3, 
                            iou=0.7, 
                            persist=True, 
                            show=True)  # Perform pose estimation

            # Check each detection in the results
            for r in results:
                # Draw keypoints and bounding boxes on the frame
                img_annotated = r.plot(boxes=True)

                # Inspect and optionally process keypoints
                keypoints = r.keypoints.xy
                for keypoint_set in keypoints:
                    for keypoint in keypoint_set:
                        x, y = keypoint.cpu().numpy()
                        if y > ground_threshold:
                            # Here you can create logic to process keypoints that are below the threshold
                            # For example, you might want to highlight or mark these keypoints differently
                            pass  # Replace 'pass' with your logic

                # Show annotated frame
                cv2.imshow("Movie", img_annotated)

                # Save
                video.write(img_annotated)

                # Close window on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    # Video file
    filename_in = 'People-Walking-2.mp4'
    filename_out = 'pose-movie-out.mp4'

    # Setup model
    model = YOLO('yolov8l-pose.pt')  # Update this path with your pose estimation model

    # Execute object detection function
    pose_estimation_movie(filename_in, filename_out, model)
