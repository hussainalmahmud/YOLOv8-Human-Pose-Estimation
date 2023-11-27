import sys
import time

import cv2
import numpy as np

# import ailia

sys.path.append('../../util')
from utils import *
from utils_2 import *
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 256

DETECTION_THRESHOLD = 0.4
DETECTION_IOU = 0.45
DETECTION_SIZE = 416


def _box2cs(box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    input_size = (IMAGE_SIZE, IMAGE_SIZE)
    x, y, w, h = box[:4]

    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def _xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0] - 1
    bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1] - 1

    return bbox_xyxy


def preprocess(img, bbox):
    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    c, s = _box2cs(bbox)
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    img = cv2.warpAffine(
        img,
        trans, (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    # normalize
    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    img_metas = [{
        'center': c,
        'scale': s,
    }]

    return img, img_metas


def postprocess(output, img_metas):
    """Decode keypoints from heatmaps.

    Args:
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
    """
    batch_size = len(img_metas)

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']

    preds, maxvals = keypoints_from_heatmaps(output, c, s)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}
    result['preds'] = all_preds
    result['boxes'] = all_boxes

    return result

def pose_estimate(net, det_net, img):
    h, w = img.shape[:2]
    n = args.max_num

    logger.debug(f'input image shape: {img.shape}')

    if det_net:
        det_net.set_input_shape(DETECTION_SIZE, DETECTION_SIZE)
        det_net.compute(img, args.threshold, args.iou)
        count = det_net.get_object_count()

        if 0 < count:
            a = sorted([
                det_net.get_object(i) for i in range(count)
            ], key=lambda x: x.prob, reverse=True)
            a = a[:n] if n else a
            bboxes = np.array([
                (int(w * obj.x), int(h * obj.y), int(w * obj.w), int(h * obj.h))
                for obj in a[:n]
            ])
        else:
            bboxes = np.array([[0, 0, w, h]])
    else:
        bboxes = np.array([[0, 0, w, h]])

    img_0 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    pose_results = []
    for bbox in bboxes:
        img, img_metas = preprocess(img_0, bbox)

        # inference
        output = net.predict([img])
        heatmap = output[0]

        result = postprocess(heatmap, img_metas)
        pose = result['preds'][0]

        # plot result
        pose_results.append({
            'bbox': _xywh2xyxy(bbox),
            'keypoints': pose,
        })

    return pose_results

# def pose_estimate(net, det_net, img):
#     h, w = img.shape[:2]

#     # Perform object detection
#     results = det_net(img)

#     # Process detection results
#     bboxes = []
#     for det in results:
#         if len(det) == 6:  # Check if detection format is [x1, y1, x2, y2, conf, cls]
#             x1, y1, x2, y2, conf, cls = det
#             if conf >= DETECTION_THRESHOLD:
#                 # Convert to format [x, y, w, h]
#                 bbox = [x1.item(), y1.item(), (x2 - x1).item(), (y2 - y1).item()]
#                 bboxes.append(bbox)

#     img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     pose_results = []

#     for bbox in bboxes:
#         img_cropped, img_metas = preprocess(img_0, bbox)

#         # Perform pose estimation (adjust based on YOLOv8l-pose output format)
#         output = net(img_cropped)
#         # Process output to get pose predictions
#         pose = process_pose_output(output)  # Implement this function based on YOLOv8l-pose output

#         pose_results.append({
#             'bbox': _xywh2xyxy(bbox),
#             'keypoints': pose,
#         })

#     return pose_results





def vis_pose_result(img, result):
    palette = np.array([
        [255, 128, 0], [255, 153, 51], [255, 178, 102],
        [230, 230, 0], [255, 153, 255], [153, 204, 255],
        [255, 102, 255], [255, 51, 255], [102, 178, 255],
        [51, 153, 255], [255, 153, 153], [255, 102, 102],
        [255, 51, 51], [153, 255, 153], [102, 255, 102],
        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
        [255, 255, 255]
    ])
    skeleton = [[1, 2], [1, 3], [2, 4], [1, 5], [2, 5], [5, 6], [6, 8],
                [7, 8], [6, 9], [9, 13], [13, 17], [6, 10], [10, 14],
                [14, 18], [7, 11], [11, 15], [15, 19], [7, 12], [12, 16],
                [16, 20]]
    
    #pose_limb_color = palette[[0] * 20]
    #pose_kpt_color = palette[[0] * 20]

    pose_limb_color = palette
    pose_kpt_color = palette
    
    img = show_result(
        img,
        result,
        skeleton,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        thickness=3)

    return img


# ======================
# Main functions
# ======================

from ultralytics.utils.files import increment_path
from pathlib import Path

def recognize_from_video(net, det_net):
    source = 'People-Walking-2.mp4'  # Make sure this path is correct
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        logger.error(f'Failed to open video file: {source}')
        return

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    save_dir = increment_path(Path('output') / 'exp', exist_ok=True)
    output_path = str(save_dir / f'{Path(source).stem}.mp4')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose_estimate(net, det_net, img)
        frame = vis_pose_result(frame, pose_results)

        cv2.imshow('frame', frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Video processing completed successfully.')


def main():
    from ultralytics import YOLO
    det_net = YOLO('yolov8l.pt')
    net = YOLO('yolov8l-pose.pt')
    recognize_from_video(net, det_net)


if __name__ == '__main__':
    main()
