from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse

def linear_interpolate(p1, p2, alpha):
    """ Linearly interpolate between points p1 and p2. """
    x = p1[0] * (1 - alpha) + p2[0] * alpha
    y = p1[1] * (1 - alpha) + p2[1] * alpha
    return (x, y)


def predict_position(history, max_history=30):
    """ Predict the current position based on history. """
    if len(history) < 2:
        return history[-1]  # Not enough data to predict
    else:
        # Predict next position based on the last two known positions
        return linear_interpolate(history[-2], history[-1], alpha=1.5)


def pose_estimation(model, source, output, view_img=False, save_img=False):
    
    model = YOLO(model)
    
    cap = cv2.VideoCapture(source)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output, fourcc, fps, (w, h))

    # Initialize tracking history
    track_history = {}
    max_lost_frames = 30  # Maximum number of frames to track after losing an object
    lost_frames_count = {}  # Dictionary to count lost frames for each track_id

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Object detection and tracking
            results = model.track(frame, 
                                save=True, 
                                # conf=0.25, 
                                iou=0.7, 
                                persist=True
                                )

            img_annotated = results[0].plot(boxes=True)

            current_track_ids = set()  # Track IDs detected in the current frame
            # Process tracking lines
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    current_track_ids.add(track_id)
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 

                    # Get or create track history for this track_id
                    track = track_history.get(track_id, [])
                    track.append((float(bbox_center[0]), float(bbox_center[1])))

                    if len(track) > 30:
                        # track.pop(0)
                        track = track[-30:]
                        
                    track_history[track_id] = track

                    # Draw tracking line
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_annotated, [points], isClosed=False, color=(0, 0, 255), thickness=2)
                
                for track_id, history in track_history.items():
                    if track_id not in current_track_ids:
                        lost_frames_count[track_id] = lost_frames_count.get(track_id, 0) + 1
                        if lost_frames_count[track_id] <= max_lost_frames and history:
                            predicted_position = predict_position(history)
                            track_history[track_id].append(predicted_position)
                            # Optionally: draw predicted position with a different color
                            cv2.circle(img_annotated, tuple(np.int32(predicted_position)), 5, (255, 0, 0), -1)
                    else:
                        lost_frames_count[track_id] = 0  # Reset lost frame count if track is found

                # Optionally: Remove old tracks
                track_history = {tid: hist for tid, hist in track_history.items() if lost_frames_count.get(tid, 0) <= max_lost_frames}

            if view_img:
                cv2.imshow("Video Pose Estimation", img_annotated)

            if save_img:
            # video_writer.write(frame)
                video.write(img_annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return



def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8l-pose.pt', help='initial weights path')
    parser.add_argument('--source', type=str,default='People-Walking-2.mp4' , help='video file path')
    parser.add_argument('--output', type=str,default='out-movie.mp4' , help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


def main(opt):
    """Main function."""
    pose_estimation(**vars(opt))

if __name__ == '__main__':

    opt = parse_opt()

    main(opt)

