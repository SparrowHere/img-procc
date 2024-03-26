import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.imgproc import log_transform, histogram_equalization

plt.style.use("seaborn-v0_8-paper")

# Q1 | Reading Images + Creating Histograms
Path: str = "/home/sparrow/cv/images"
img_gray = cv2.imread(Path + "/gray9.jpg", cv2.IMREAD_GRAYSCALE)
img_dark = cv2.imread(Path + "/dark9.jpg", cv2.IMREAD_GRAYSCALE)

'''
cv2.imshow("Dark Image", img_dark)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Getting histogram/s of the image/s
plt.hist(
    img_dark.ravel(),
    256,
    [0, 256],
    label = "Dark Image",
);
plt.title("Histogram of the Dark Image");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

plt.hist(
    img_gray.ravel(),
    256,
    [0, 256],
    label = "Gray Image",
);
plt.title("Histogram of the Gray Image");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();
'''
# Q1 | Log Transform
'''
def log_transform(image: np.ndarray, c: float = 1) -> np.ndarray:
    """
    Apply log transformation to an image.

    Parameters:
        image (`numpy.ndarray`): Input image.
        c (`float`): Constant multiplier for adjusting the transformation (default is 1).

    Returns:
        `numpy.ndarray`: Transformed image.
    """
    # s = c * log(1 + r)
    transformed_image = c * np.log(1 + image.astype(np.float32))

    # Normalizing the pixel values to the range [0, 255]
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

    return transformed_image
'''    
dark_transformed = log_transform(img_dark, 40)

cv2.imshow("Dark Image", img_dark)
cv2.imshow("Log Dark Image", dark_transformed)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Q1 | Histogram Equalization (OpenCV)
'''
cv2.equalizeHist(img_dark, img_dark)

cv2.imshow("Dark Equalized Image", img_dark)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(
    img_dark.ravel(),
    256,
    [0, 256],
    label = "Dark Image",
);
plt.title("Dark Equalized Image");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();
'''