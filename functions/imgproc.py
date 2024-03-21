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

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to the input image.

    Parameters:
        image (`np.ndarray`): Input image (numpy array).
        gamma (`float`): Gamma value for correction.

    Returns:
        `np.ndarray`: Corrected image.
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    corrected_image = cv2.LUT(image, table)

    return corrected_image