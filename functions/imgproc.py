import numpy as np
from scipy.ndimage import median_filter
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

def normalize(image: np.ndarray, encoding: np.int) -> np.ndarray:
    """Normalizes image based on `encoding`

    Parameters:
        image (np.ndarray): Image to be normalized
        encoding (np.int): Bit depth used in image encoding

    Returns:
        np.ndarray: Normalized image
    """
    return encoding * (image - image.min() / (image.max() - image.min()))

def log_transform(image: np.ndarray, c: float = 1) -> np.ndarray:
    """
    Apply logarithm transform to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        c (float): Constant value for the logarithm transform (default is 1).

    Returns:
        numpy.ndarray: Transformed image.
    """
    # Apply logarithm transform
    transformed_image = c * np.log(1 + image.astype(np.float32))

    # Scale the image to 0-255 range
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

    return transformed_image

import numpy as np

def histogram_equalization(image: np.ndarray, encoding: int = 8) -> np.ndarray:
    """
    Perform histogram equalization on the input grayscale image.

    Parameters:
        image (`numpy.ndarray`): Input **grayscale** image (2D array).
        encoding (`ìntt`): Encoding used for the image.

    Returns:
        `numpy.ndarray`: Equalized image.
    """
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Calculate cumulative distribution function (CDF) of the histogram
    CDF = hist.cumsum()

    cdf_norm = ((CDF - CDF.min()) * 2**encoding) / (CDF.max() - CDF.min())

    equalized_img = np.interp(image.flatten(), bins[:-1], cdf_norm)

    # Reshape the equalized image to the original shape
    equalized_img = equalized_img.reshape(image.shape)

    return equalized_img.astype(np.uint8)
