import cv2
import numpy as np




def preprocess_image(image, model_img_size=640):
    """Preprocesses an image to be compatible with the model.

    Args:
        image (_type_) : should be a valid image file path or a numpy array
        model_img_size (int, optional): _description_. Defaults to 640.

    Returns: image_padded (numpy array): a numpy array of the image
    """
    # Calculate the ratio of the target dimensions to the original dimensions
    h, w = image.shape[:2]
    scale = model_img_size / max(h, w)
    
    # Scale the image dimensions to the model's expected size
    h_scaled, w_scaled = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (w_scaled, h_scaled))

    # Calculate padding to maintain aspect ratio
    top_pad = (model_img_size - h_scaled) // 2
    bottom_pad = model_img_size - h_scaled - top_pad
    left_pad = (model_img_size - w_scaled) // 2
    right_pad = model_img_size - w_scaled - left_pad

    # Apply padding
    image_padded = cv2.copyMakeBorder(image_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image_padded

# sharpen image

def sharpen_image(image):
    """Sharpen an image

    Args:
        image (numpy array): a numpy array of the image

    Returns:
        image_sharpened (numpy array): a numpy array of the image
    """
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    image_sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return image_sharpened