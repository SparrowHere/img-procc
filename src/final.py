import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

class Processing:
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
    def gaussian_blur(image: np.ndarray, kernel_size: tuple, sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            kernel_size (tuple): Size of the Gaussian kernel (width, height).
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: Blurred image.
        """
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
        return blurred_image

    @staticmethod
    def fft(image: np.ndarray) -> np.ndarray:
        """
        Compute the Fast Fourier Transform (FFT) of the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).

        Returns:
            np.ndarray: FFT of the input image.
        """
        fft_image = np.fft.fft2(image)
        fft_shifted_image = np.fft.fftshift(fft_image)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shifted_image))
        return magnitude_spectrum

    @staticmethod
    def fft_filter(image: np.ndarray, filter_type: str, cutoff_frequency: float, order: float) -> np.ndarray:
        """
        Apply frequency domain filtering to the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            filter_type (str): Type of filter ('low-pass' or 'high-pass').
            cutoff_frequency (float): Cutoff frequency of the filter.
            order (float): Order of the filter.

        Returns:
            np.ndarray: Filtered image.
        """
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows // 2 , cols // 2
        
        mask = np.zeros((rows, cols), np.uint8)
        
        if filter_type == 'low-pass':
            mask[int(crow - cutoff_frequency):int(crow + cutoff_frequency), int(ccol - cutoff_frequency):int(ccol + cutoff_frequency)] = 1
        elif filter_type == 'high-pass':
            mask[int(crow - cutoff_frequency):int(crow + cutoff_frequency), int(ccol - cutoff_frequency):int(ccol + cutoff_frequency)] = 0
            mask = 1 - mask
        
        f_transform_shifted = f_transform_shifted * mask
        f_transform = np.fft.ifftshift(f_transform_shifted)
        filtered_image = np.fft.ifft2(f_transform)
        filtered_image = np.abs(filtered_image)
        return filtered_image

    @staticmethod
    def edge_sharpening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply edge sharpening to the input image using a given kernel.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            kernel (np.ndarray): Sharpening kernel.

        Returns:
            np.ndarray: Sharpened image.
        """
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image

    @staticmethod
    def plot_histogram(image: np.ndarray, title: str) -> None:
        """
        Plot the histogram of the input image.

        Parameters:
            image (np.ndarray): Input image (numpy array).
            title (str): Title for the histogram plot.
        """
        plt.hist(image.ravel(), 256, [0, 256])
        plt.title(title)
        plt.xlabel("Gray Level")
        plt.ylabel("Count")
        plt.show()