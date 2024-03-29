import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
plt.style.use("seaborn-v0_8-paper")

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

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization on the input grayscale image.

    Parameters:
        image (`numpy.ndarray`): Input grayscale image (2D array).

    Returns:
        `numpy.ndarray`: Equalized image.
    """
    # Calculate histogram
    hist, bins = np.histogram(
        image.flatten(),
        bins = 256,
        range = [0, 255]
    )

    # Calculate "Cumulative Distribution Function (CDF)"
    CDF = hist.cumsum()

    # Normalize the CDF to the range [0, 255]
    CDF_norm = ((CDF - CDF.min()) * 255) / (CDF.max() - CDF.min())

    # Interpolate the CDF values for each pixel intensity
    equalized_image = np.interp(
        image.ravel(),
        bins[:-1],
        CDF_norm
    )
    equalized_image = equalized_image.reshape(image.shape)

    return equalized_image.astype(np.uint8)

# Read the image/s
Path: str = "/home/sparrow/cv/images"
img_gray: np.ndarray = cv2.imread(Path + "/gray9.jpg")
img_dark: np.ndarray = cv2.imread(Path + "/dark9.jpg")
img_shadow: np.ndarray = cv2.imread(Path + "/shadow9.jpg")
img_nn: np.ndarray = cv2.imread(Path + "/9nn.jpg")
img_sp: np.ndarray = cv2.imread(Path + "/9sp.jpg")

# Convert images to Grayscale U8
img_dark = cv2.cvtColor(img_dark, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
img_shadow = cv2.cvtColor(img_shadow, cv2.COLOR_RGB2GRAY)
img_nn = cv2.cvtColor(img_nn, cv2.COLOR_RGB2GRAY)
img_sp = cv2.cvtColor(img_sp, cv2.COLOR_RGB2GRAY)

# Getting histogram of the image
plt.hist(
    img_dark.ravel(),
    256,
    [0, 256],
    color = "#1f77b4",
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
    color = "#ff7f0e",
    label = "Gray Image",
);
plt.title("Histogram of the Gray Image");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

# Gamma correction of an image/s
img_dark_corrected = gamma_correction(img_dark, 0.5)
cv2.imshow("Gamma Corrected Image (gamma = 0.5)", img_dark_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(
    img_dark_corrected.ravel(),
    256,
    [0, 256],
    label = "Corrected Dark Image",
);
plt.title("Gamma Corrected Image (Gamma = 0.5)");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

# Log-Transform of an image/s 
dark_transformed = log_transform(img_dark, 40)

cv2.imshow("Dark Image", img_dark)
cv2.imshow("Log-transformed Dark Image", dark_transformed)
cv2.waitKey(5000)
cv2.destroyAllWindows()

plt.hist(
    dark_transformed.ravel(),
    256,
    [0, 256],
    label = "Transformed Dark Image",
);
plt.title("Log-Transformed Dark Image (C = 40)");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

# Equalize histogram
img_dark_eq = cv2.equalizeHist(img_dark)

cv2.imshow("Dark Equalized Image", img_dark_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot histogram
plt.hist(
    img_dark_eq.ravel(),
    256,
    [0, 256],
    label = "Dark Image",
);
plt.title("Dark Equalized Image");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

# Equalize histogram (with custom function)
img_dark_eq = histogram_equalization(img_dark)

cv2.imshow("Dark Equalized Image", img_dark_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(
    img_dark_eq.ravel(),
    256,
    [0, 256],
    label = "Dark Image",
);
plt.title("Dark Equalized Image (from Scratch)");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();

# Get Gaussian-Blurred Image (for shadow restoration)
img_shadow_gauss = cv2.GaussianBlur(
    img_shadow,
    (75, 75),
    sigmaX = 128,
    sigmaY = 128
)

cv2.imshow("Image with Gaussian Blur", img_shadow_gauss)

img_shadow_log = np.log(1 + img_shadow_gauss.astype(np.float32))
img_shadow_gauss = 255 * (img_shadow_log / img_shadow_log.max())
img_shadow_log_gauss = np.uint8(img_shadow_gauss)

cv2.imshow("Image with Log-Gaussian Blur", img_shadow_log_gauss)

img_shadow_removed = img_shadow - img_shadow_log_gauss

cv2.imshow("New Image", img_shadow_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Spatial Filtering
img_nn_box = cv2.blur(img_nn, (3, 3))   # Box Filtering
img_nn_gauss = cv2.GaussianBlur(    # Gauss Filtering
    img_nn,
    (5, 5),
    sigmaX = 1,
    sigmaY = 1
)
img_nn_mdn = cv2.medianBlur(    # Median Filtering
    img_nn,
    5
)

img_sp_box = cv2.blur(img_sp, (3, 3))   # Box Filtering
img_sp_gauss = cv2.GaussianBlur(    # Gauss Filtering
    img_sp,
    (5, 5),
    sigmaX = 1,
    sigmaY = 1
)
img_sp_mdn = cv2.medianBlur(    # Median Filtering
    img_sp,
    3
)

# # Save ROI image to compare the details
X, Y, H, W = 200, 50, 50, 50
img_sp_mdn_roi = cv2.cvtColor(img_sp_mdn, cv2.COLOR_GRAY2BGR)
cv2.rectangle(
    img_sp_mdn_roi,
    (X, Y),
    (X + W, Y + H),
    (180, 119, 31),
    2
)
img_sp_mdn_cropped = img_sp_mdn[Y:Y + H, X:X + W]

cv2.imshow("New Image", img_sp_mdn_roi)
cv2.imshow("Cropped Image", img_sp_mdn_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Frequency Domain
img_nn_f = np.fft.fft2(np.float32(img_nn))
img_nn_fs = np.fft.fftshift(img_nn_f)
img_nn_fsm = 20 * np.log(np.abs(img_nn_fs))

# Ideal Low-Pass Filter
H = np.zeros_like(img_nn)

D0 = 25    # Radius of the filter
CX = H.shape[1] // 2
CY = H.shape[0] // 2
cv2.circle(H, (CX, CY), D0, (255, 255, 255), -1)
            
cv2.imshow("Ideal Low-Pass Filter", H)

img_nn_ideal = np.multiply(img_nn_fs, H) / 255
img_nn_ideal_s = np.fft.ifftshift(img_nn_ideal)
img_nn_ideal_si = np.fft.ifft2(img_nn_ideal_s)
img_nn_ideal_sin = np.abs(img_nn_ideal_si).clip(0, 255).astype(np.uint8)

cv2.imshow("Filtered Image (Normalized)", img_nn_ideal_sin)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Frequency Domain
img_nn_f = np.fft.fft2(np.float32(img_nn)) / (img_nn.shape[0] * img_nn.shape[1])
img_nn_fs = np.fft.fftshift(img_nn_f)
img_nn_fsm = 20 * np.log(np.abs(img_nn_fs))

cv2.imshow("Image in Frequency Domain", img_nn_fsm)

# Butterworth Filter
img_nn_bttr = filters.butterworth(
    img_nn,
    cutoff_frequency_ratio = 0.005,
    order = 0.35,
    high_pass = False,
    squared_butterworth = True,
    npad = 0
)
img_nn_bttr_n = np.uint8(255 * ((img_nn_bttr - img_nn_bttr.min()) / (img_nn_bttr.max() - img_nn_bttr.min())))

img_nn_bttr_n_f = np.fft.fft2(img_nn_bttr_n) / (img_nn_bttr_n.shape[0] * img_nn_bttr_n.shape[1])
img_nn_bttr_n_fs = np.fft.fftshift(img_nn_bttr_n_f)
img_nn_bttr_n_fsm = 20 * np.log(np.abs(img_nn_bttr_n_fs))

cv2.imshow("Filtered Image in Frequency Domain", img_nn_bttr_n_fsm)

cv2.imshow("Original Image", img_nn)
cv2.imshow("Filtered Image", img_nn_bttr_n)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sharpening Kernel
K = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
img_gray_sharp = cv2.filter2D(img_gray, -1, K)

# Canny Filter
img_gray_canny = cv2.Canny(img_gray, 50, 150)

cv2.imshow("Original Image", img_gray)
cv2.imshow("Filtered Image (Sharpen)", img_gray_sharp)
cv2.imshow("Filtered Image (Canny)", img_gray_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()