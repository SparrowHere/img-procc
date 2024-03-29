# ======================================= #
# AUTH: Ahmet B. SERCE
# NO: 19332629022
# ======================================= #

#%% # Definition of `Processing` Class
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import butterworth

plt.style.use("seaborn-v0_8-paper")

class Processing:
    @staticmethod
    def plot_histogram(image: np.ndarray, color="#1f77b4", title="Histogram", xlabel="Gray Level", ylabel="Count", label="Image") -> None:
        """
        Plot the histogram of an image.

        Parameters:
            image (np.ndarray): Input image.
            title (str): Title of the plot (default is "Histogram").
            xlabel (str): Label for the x-axis (default is "Gray Level").
            ylabel (str): Label for the y-axis (default is "Count").
            color (str): Color of the histogram bars (default is "#1f77b4").
            label (str): Label for the histogram (default is "Image").
        """
        plt.hist(
            image.ravel(),
            256,
            [0, 256],
            color=color,
            label = label,
        );
        plt.title(title);
        plt.xlabel(xlabel);
        plt.ylabel(ylabel);
        plt.show();
        
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            gamma (float): Gamma value for correction.

        Returns:
            np.ndarray: Corrected image.
        """
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_image = cv2.LUT(image, table)
        
        return corrected_image

    @staticmethod
    def log_transform(image: np.ndarray, c: float = 1) -> np.ndarray:
        """
        Apply log transformation to an image.

        Parameters:
            image (numpy.ndarray): Input image.
            c (float): Constant multiplier for adjusting the transformation (default is 1).

        Returns:
            numpy.ndarray: Transformed image.
        """
        transformed_image = c * np.log(1 + image.astype(np.float32))
        transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
        return transformed_image

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Perform histogram equalization on the input grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image (2D array).

        Returns:
            numpy.ndarray: Equalized image.
        """
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 255])
        CDF = hist.cumsum()
        CDF_norm = ((CDF - CDF.min()) * 255) / (CDF.max() - CDF.min())
        equalized_image = np.interp(image.ravel(), bins[:-1], CDF_norm)
        equalized_image = equalized_image.reshape(image.shape)
        return equalized_image.astype(np.uint8)
    
    @staticmethod

    def ideal_LP_FFT(image: np.ndarray, cutoff_frequency: float) -> np.ndarray:
        """
        Apply ideal low-pass filter to the input image in frequency domain.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            cutoff_frequency (float): Cutoff frequency of the filter.

        Returns:
            np.ndarray: Filtered image.
        """
        # Compute FFT of the image
        f_transform = np.fft.fft2(np.float32(image))
        f_transform_shifted = np.fft.fftshift(f_transform)
                
        # Compute center points
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create ideal low-pass filter mask
        mask = np.zeros((rows, cols), np.uint8)
        mask[int(crow - cutoff_frequency):int(crow + cutoff_frequency), 
            int(ccol - cutoff_frequency):int(ccol + cutoff_frequency)] = 1
        
        # Apply filter
        f_transform_shifted = np.multiply(f_transform_shifted, mask)
        
        # Compute inverse FFT of the filtered image
        f_transform = np.fft.ifftshift(f_transform_shifted)
        filtered_image = np.fft.ifft2(f_transform)
        filtered_image = np.abs(filtered_image).clip(0, 255).astype(np.uint8)
        
        return filtered_image


    @staticmethod
    def gaussian_FFT(image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian filter to the input image in frequency domain.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: Filtered image.
        """
        # Compute FFT of an Image
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # Compute Center Points
        rows, cols = image.shape
        crow, ccol = rows // 2 , cols // 2
        
        x = np.arange(cols) - crow
        y = np.arange(rows) - ccol
        X, Y = np.meshgrid(x, y)
        
        # Create Gaussian Filter
        gaussian_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # Apply Filter
        f_transform_shifted = np.multiply(f_transform_shifted, gaussian_filter)
        
        # Compute IFFT of an Image
        f_transform = np.fft.ifftshift(f_transform_shifted)
        filtered_image = np.fft.ifft2(f_transform)
        filtered_image = np.abs(filtered_image).clip(0, 255).astype(np.uint8)
        
        return filtered_image

    @staticmethod
    def butterworth_FFT(img: np.ndarray, cutoff_frequency_ratio: float, order: float) -> tuple[np.ndarray]:
        """
        Apply Butterworth filter to the input image in frequency domain.

        Parameters:
            img (np.ndarray): Input image (numpy array).
            cutoff_frequency_ratio (float): Cutoff frequency ratio of the filter.
            order (float): Order of the filter.

        Returns:
            np.ndarray: Filtered image.
        """
        # Compute FFT of the image
        img_f = np.fft.fft2(np.float32(img)) / (img.shape[0] * img.shape[1])
        img_fs = np.fft.fftshift(img_f)
        img_fsm = 20 * np.log(np.abs(img_fs))

        # Apply Butterworth filter
        img_bttr = butterworth(
            img,
            cutoff_frequency_ratio=cutoff_frequency_ratio,
            order=order,
            high_pass=False,
            squared_butterworth=True,
            npad=0
        )
        img_bttr_n = np.uint8(255 * ((img_bttr - img_bttr.min()) / (img_bttr.max() - img_bttr.min())))

        # Compute FFT of the filtered image
        img_bttr_n_f = np.fft.fft2(img_bttr_n) / (img_bttr_n.shape[0] * img_bttr_n.shape[1])
        img_bttr_n_fs = np.fft.fftshift(img_bttr_n_f)
        img_bttr_n_fsm = 20 * np.log(np.abs(img_bttr_n_fs))

        return img_bttr_n_fsm, img_bttr_n

    @staticmethod
    def laplacian(image: np.ndarray) -> np.ndarray:
        """
        Apply Laplacian filter to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).

        Returns:
            np.ndarray: Filtered image.
        """
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        filtered_image = cv2.filter2D(image, -1, laplacian_kernel)
        return filtered_image

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).

        Returns:
            np.ndarray: Filtered image.
        """
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_image = cv2.filter2D(image, -1, sharpen_kernel)
        return filtered_image

    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply median blur to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            kernel_size (int): Size of the median blur kernel.

        Returns:
            np.ndarray: Filtered image.
        """
        filtered_image = cv2.medianBlur(image, kernel_size)
        return filtered_image

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: tuple, sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            kernel_size (tuple): Size of the Gaussian kernel (rows, cols).
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: Filtered image.
        """
        filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return filtered_image

#%% Preparation

Path: str = "/home/sparrow/cv/images"
img_gray: np.ndarray = cv2.imread(Path + "/gray9.jpg")
img_dark: np.ndarray = cv2.imread(Path + "/dark9.jpg")
img_shadow: np.ndarray = cv2.imread(Path + "/shadow9.jpg")
img_nn: np.ndarray = cv2.imread(Path + "/9nn.jpg")
img_sp: np.ndarray = cv2.imread(Path + "/9sp.jpg")

img_dark = cv2.cvtColor(img_dark, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
img_shadow = cv2.cvtColor(img_shadow, cv2.COLOR_RGB2GRAY)
img_nn = cv2.cvtColor(img_nn, cv2.COLOR_RGB2GRAY)
img_sp = cv2.cvtColor(img_sp, cv2.COLOR_RGB2GRAY)
#%% Q1a | Histograms and Restoration Operations

Processing.plot_histogram(img_gray, "#ff7f0e", "Histogram of the Gray Image")
Processing.plot_histogram(img_dark, "#1f77b4", "Histogram of the Gray Image")
# %% Q1a | Restoration - Gamma Correction

img_dark_corrected = Processing.gamma_correction(img_dark, 0.5)
cv2.imshow("Gamma Corrected Image (gamma = 0.5)", img_dark_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

Processing.plot_histogram(img_dark_corrected, "#1f77b4", "Gamma Corrected Image (Gamma = 0.5)")

# %% Q1a | Restoration - Logarithm Transform

dark_transformed = Processing.log_transform(img_dark, 40)

cv2.imshow("Dark Image", img_dark)
cv2.imshow("Log-transformed Dark Image", dark_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()

Processing.plot_histogram(dark_transformed, "#1f77b4", "Log-Transformed Dark Image (C = 40)")

# %% Q1b | Histogram Equalization (OpenCV)

img_dark_eq = cv2.equalizeHist(img_dark)

cv2.imshow("Dark Equalized Image", img_dark_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

Processing.plot_histogram(img_dark_eq, "#1f77b4", "Dark Equalized Image (OpenCV)")
# %% Q1b | Histogram Equalization (from Scratch)

img_dark_eq = Processing.histogram_equalization(img_dark)

cv2.imshow("Dark Equalized Image", img_dark_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

Processing.plot_histogram(img_dark_eq, "#1f77b4", "Dark Equalized Image (from Scratch)")
# %% Q1c | Shadowy Image Restoration

img_shadow_gauss = Processing.gaussian_blur(img_shadow, (121, 121), 128)

img_shadow_log = np.log(1 + img_shadow_gauss.astype(np.float32))
img_shadow_gauss = 255 * (img_shadow_log / img_shadow_log.max())
img_shadow_log_gauss = np.uint8(img_shadow_gauss)

cv2.imshow("Image with Log-Gaussian Blur", img_shadow_log_gauss)
cv2.imshow("Image with Gaussian Blur", img_shadow_gauss)
cv2.waitKey(0)

# Substracting the blurred image from the original image
img_shadow_removed = img_shadow - img_shadow_log_gauss

cv2.imshow("New Image", img_shadow_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% Q1d | Noise Reduction with Spatial Filtering

img_nn_box = cv2.blur(img_nn, (3, 3))
img_nn_gauss = Processing.gaussian_blur(img_nn, (5, 5), 1)
img_nn_mdn = Processing.median_blur(img_nn, 3)

img_sp_box = cv2.blur(img_nn, (3, 3))
img_sp_gauss = Processing.gaussian_blur(img_nn, (5, 5), 1)
img_sp_mdn = Processing.median_blur(img_nn, 3)


# Create ROI and save it to compare
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
# %% Q1e | Noise Reduction in Frequency Domain

img_nn_ideal = Processing.ideal_LP_FFT(img_nn, 30)
img_sp_ideal = Processing.ideal_LP_FFT(img_sp, 30)

img_nn_gaussian = Processing.gaussian_FFT(img_nn, 20)   # ( !1! )
img_sp_gaussian = Processing.gaussian_FFT(img_sp, 20)

img_nn_bttr = Processing.butterworth_FFT(img_nn, 0.005, 0.35)[1]
img_sp_bttr = Processing.butterworth_FFT(img_sp, 0.005, 0.35)[1]

cv2.imshow("Filtered Image NN (Ideal LP)", img_nn_ideal)
cv2.imshow("Filtered Image SP (Ideal LP)", img_sp_ideal)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Filtered Image NN (Gaussian LP)", img_nn_gaussian)
cv2.imshow("Filtered Image SP (Gaussian LP)", img_sp_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Filtered Image NN (Butterworth LP)", img_nn_bttr)
cv2.imshow("Filtered Image SP (Butterworth LP)", img_sp_bttr)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %% Q1f | Bringing-out Edges

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
# %%
