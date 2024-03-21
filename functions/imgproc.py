import numpy as np
import cv2

def bitPlaneSlicing(img: np.ndarray, depth: int, dtype: type = np.uint8) -> np.ndarray:
    """
    Perform bit-plane slicing on the input image.

    Parameters:
        img (`numpy.ndarray`): The input image.
        depth (`int`): The bit depth for slicing.
        dtype (`type`, optional): The data type for the output image. Default is `np.uint8`.

    Returns:
        `numpy.ndarray`: The sliced image with the specified data type.
    """
    mask: np.ndarray = np.ones(img.shape, dtype = dtype) * ((1 << depth) - 1)
    sliced: np.ndarray = cv2.Mat(np.zeros(img.shape, dtype = dtype))
    cv2.bitwise_and(img, mask, sliced)
    
    return sliced