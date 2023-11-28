import os
import sys
from logging import getLogger

import cv2
import numpy as np

logger = getLogger(__name__)


def imread(filename, flags=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        logger.error(f"File does not exist: {filename}")
        sys.exit()
    data = np.fromfile(filename, np.int8)
    img = cv2.imdecode(data, flags)
    return img


def normalize_image(image, normalize_type='255'):
    """
    Normalize image

    Parameters
    ----------
    image: numpy array
        The image you want to normalize
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    normalized_image: numpy array
    """
    if normalize_type == 'None':
        return image
    elif normalize_type == '255':
        return image / 255.0
    elif normalize_type == '127.5':
        return image / 127.5 - 1.0
    elif normalize_type == 'ImageNet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    else:
        logger.error(f'Unknown normalize_type is given: {normalize_type}')
        sys.exit()


def load_image(
        image_path,
        image_shape,
        rgb=True,
        normalize_type='255',
        gen_input_ailia=False,
):
    """
    Loads the image of the given path, performs the necessary preprocessing,
    and returns it.

    Parameters
    ----------
    image_path: string
        The path of image which you want to load.
    image_shape: (int, int)  (height, width)
        Resizes the loaded image to the size required by the model.
    rgb: bool, default=True
        Load as rgb image when True, as gray scale image when False.
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': output range: 0 and 1
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet.
        - 'None': no normalization
    gen_input_ailia: bool, default=False
        If True, convert the image to the form corresponding to the ailia.

    Returns
    -------
    image: numpy array
    """
    # rgb == True --> cv2.IMREAD_COLOR
    # rbg == False --> cv2.IMREAD_GRAYSCALE
    image = imread(image_path, int(rgb))
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, normalize_type)
    image = cv2.resize(image, (image_shape[1], image_shape[0]))

    if gen_input_ailia:
        if rgb:
            image = image.transpose((2, 0, 1))  # channel first
            image = image[np.newaxis, :, :, :]  # (batch_size, channel, h, w)
        else:
            image = image[np.newaxis, np.newaxis, :, :]
    return image


def get_image_shape(image_path):
    tmp = imread(image_path)
    height, width = tmp.shape[0], tmp.shape[1]
    return height, width


# (ref: https://qiita.com/yasudadesu/items/dd3e74dcc7e8f72bc680)
def draw_texts(img, texts, font_scale=0.7, thickness=2):
    h, w, c = img.shape
    offset_x = 10
    initial_y = 0
    dy = int(img.shape[1] / 15)
    color = (0, 0, 0)  # black

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        offset_y = initial_y + (i+1)*dy
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_result_on_img(img, texts, w_ratio=0.35, h_ratio=0.2, alpha=0.4):
    overlay = img.copy()
    pt1 = (0, 0)
    pt2 = (int(img.shape[1] * w_ratio), int(img.shape[0] * h_ratio))

    mat_color = (200, 200, 200)
    fill = -1
    cv2.rectangle(overlay, pt1, pt2, mat_color, fill)

    mat_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    draw_texts(mat_img, texts)
    return mat_img


import os
import sys
from logging import getLogger

# import ailia
import cv2
import numpy as np
import json

logger = getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
# from image_utils import imread  # noqa: E402


def preprocessing_img(img):
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f'{image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def letterbox_convert(frame, det_shape):
    """
    Adjust the size of the frame from the webcam to the ailia input shape.

    Parameters
    ----------
    frame: numpy array
    det_shape: tuple
        ailia model input (height,width)

    Returns
    -------
    resized_img: numpy array
        Resized `img` as well as adapt the scale
    """
    height, width = det_shape[0], det_shape[1]
    f_height, f_width = frame.shape[0], frame.shape[1]
    scale = np.max((f_height / height, f_width / width))

    # padding base
    img = np.zeros(
        (int(round(scale * height)), int(round(scale * width)), 3),
        np.uint8
    )
    start = (np.array(img.shape) - np.array(frame.shape)) // 2
    img[
        start[0]: start[0] + f_height,
        start[1]: start[1] + f_width
    ] = frame
    resized_img = cv2.resize(img, (width, height))
    return resized_img


# def reverse_letterbox(detections, img, det_shape):
#     h, w = img.shape[0], img.shape[1]

#     pad_x = pad_y = 0
#     if det_shape != None:
#         scale = np.max((h / det_shape[0], w / det_shape[1]))
#         start = (det_shape[0:2] - np.array(img.shape[0:2]) / scale) // 2
#         pad_x = start[1] * scale
#         pad_y = start[0] * scale

#     new_detections = []
#     for detection in detections:
#         logger.debug(detection)
#         r = ailia.DetectorObject(
#             category=detection.category,
#             prob=detection.prob,
#             x=(detection.x * (w + pad_x * 2) - pad_x) / w,
#             y=(detection.y * (h + pad_y * 2) - pad_y) / h,
#             w=(detection.w * (w + pad_x * 2)) / w,
#             h=(detection.h * (h + pad_y * 2)) / h,
#         )
#         new_detections.append(r)

#     return new_detections


def plot_results(detector, img, category=None, segm_masks=None, logging=True):
    """
    :param detector: ailia.Detector, or list of ailia.DetectorObject
    :param img: ndarray data of image
    :param category: list of category_name
    :param segm_masks:
    :param logging: output log flg
    :return:
    """
    h, w = img.shape[0], img.shape[1]

    count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)
    if logging:
        print(f'object_count={count}')

    # prepare color data
    colors = []
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]

        # print result
        if logging:
            print(f'+ idx={idx}')
            print(
                f'  category={obj.category}[ {category[int(obj.category)]} ]'
                if not isinstance(obj.category, str) and category is not None
                else f'  category=[ {obj.category} ]'
            )
            print(f'  prob={obj.prob}')
            print(f'  x={obj.x}')
            print(f'  y={obj.y}')
            print(f'  w={obj.w}')
            print(f'  h={obj.h}')

        if isinstance(obj.category, int) and category is not None:
            color = hsv_to_rgb(256 * obj.category / (len(category) + 1), 255, 255)
        else:
            color = hsv_to_rgb(256 * idx / (len(detector) + 1), 255, 255)
        colors.append(color)

    # draw segmentation area
    if segm_masks is not None and 0 < len(segm_masks):
        for idx in range(count):
            mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, axis=2).astype(bool)
            color = colors[idx][:3]
            fill = np.repeat(np.repeat([[color]], img.shape[0], 0), img.shape[1], 1)
            img[:, :, :3][mask] = img[:, :, :3][mask] * 0.7 + fill[mask] * 0.3

    # draw bounding box
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        top_left = (int(w * obj.x), int(h * obj.y))
        bottom_right = (int(w * (obj.x + obj.w)), int(h * (obj.y + obj.h)))

        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, 4)

    # draw label
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        fontScale = w / 2048

        text = category[int(obj.category)] \
            if not isinstance(obj.category, str) and category is not None \
            else obj.category
        text = "{} {}".format(text, int(obj.prob * 100) / 100)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
        tw = textsize[0]
        th = textsize[1]

        margin = 3

        x1 = int(w * obj.x)
        y1 = int(h * obj.y)
        x2 = x1 + tw + margin
        y2 = y1 + th + margin

        # check the x,y x2,y2 are inside the image
        if x1 < 0:
            x1 = 0
        elif x2 > w:
            x1 = w - (tw + margin)

        if y1 < 0:
            y1 = 0
        elif y2 > h:
            y1 = h - (th + margin)

        # recompute x2, y2 if shift occured
        x2 = x1 + tw + margin
        y2 = y1 + th + margin

        top_left = (x1, y1)
        bottom_right = (x2, y2)

        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

        text_color = (255, 255, 255, 255)
        cv2.putText(
            img,
            text,
            (top_left[0], top_left[1] + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            text_color,
            1
        )
    return img


def write_predictions(file_name, detector, img=None, category=None, file_type='txt'):
    h, w = (img.shape[0], img.shape[1]) if img is not None else (1, 1)

    count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)

    if file_type == 'json':
        results = []
        for idx in range(count):
            obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
            label = category[int(obj.category)] \
                if not isinstance(obj.category, str) and category is not None \
                else obj.category
            prob = float(obj.prob)
            bbox_x = float(w * obj.x)
            bbox_y = float(h * obj.y)
            bbox_w = float(w * obj.w)
            bbox_h = float(h * obj.h)
            results.append({
                'category': label, 'prob': prob,
                'x': bbox_x, 'y': bbox_y, 'w': bbox_w, 'h': bbox_h
            })
        with open(file_name, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(file_name, 'w') as f:
            for idx in range(count):
                obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
                label = category[int(obj.category)] \
                    if not isinstance(obj.category, str) and category is not None \
                    else obj.category

                f.write('%s %f %d %d %d %d\n' % (
                    label.replace(' ', '_'),
                    obj.prob,
                    int(w * obj.x), int(h * obj.y),
                    int(w * obj.w), int(h * obj.h),
                ))
