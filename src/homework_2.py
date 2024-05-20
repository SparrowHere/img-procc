# %%
from typing import Sequence
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["font.family"] = "monospace"

import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
class Image:
    """
    A class to represent an image. It provides various image processing methods including various thresholding techniques, denoising, and clustering.
    
    Attributes:
        path (`str`): The path to the image file.
        image (`np.ndarray`): The original image.
        gray (`np.ndarray`): The grayscale version of the image.
        channels (`Sequence[np.ndarray]`): The channels of the image.
        
    Methods:
        filename(`str`) -> `int`: Extracts the number from the filename.
        show(`np.ndarray`, `bool`, `str`) -> `None`: Displays the image.
        subplot(`Sequence[np.ndarray]`, `Sequence[str]`, `int`, `int`, `bool`, `str`, `str`) -> `None`: Displays multiple images in a subplot.
        adaptiveThresh(`bool`, `int`) -> `np.ndarray`: Applies adaptive thresholding to the image.
        otsuThresh(`bool`, `int`) -> `np.ndarray`: Applies Otsu's thresholding to the image.
        denoise(`np.ndarray`, `str`) -> `np.ndarray`: Applies denoising to the image.
        KMeans(`bool`, `int`) -> `np.ndarray`: Applies K-Means clustering to the image.
        Snakes(`bool`) -> `np.ndarray`: Applies active contours (snakes) to the image.
    """
    def __init__(self, path: str) -> None:
        self.path = path    # Path to the image file
        self.image: np.ndarray = cv2.imread(path)   # Read the image
        self.gray: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)    # Convert the image to grayscale
        self.channels: Sequence[np.ndarray] = cv2.split(self.image)     # Split the image into channels
    
    @staticmethod
    def filename(filename: str) -> int:
        """
        Extracts the number from the filename.

        Parameters:
            filename (`str`): The filename from which to extract the number.

        Returns:
            `int`: The extracted number.

        """
        match = re.search(r'(\d+)', filename)       # Search for a number in the filename
        if match:       # If a number is found
            return int(match.group(1))      # Return the number
        else:       # If no number is found
            return -1       # Return -1
        
    def show(self, image: np.ndarray = None, gray: bool = False, title: str = 'Image') -> None: # type: ignore
        """
        Displays the image.

        Parameters:
            image (`np.ndarray`): The image to display. If None, display the original image.
            gray (`bool`): Whether to display the image in grayscale. Default is `False`.
            title (`str`): The title of the window. Default is `'Image'`.

        """
        if image is None:       # If no image is provided
            image = self.gray if gray else self.image       # Use the grayscale image if `gray` is True, otherwise use the original image
        if gray:        # If the image is in grayscale
            plt.imshow(image, cmap='gray')      # Display the image in grayscale
        else:       # If the image is in color
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))      # Display the image in color
        plt.title(title)        # Set the title of the window
        plt.show()      # Show the image

    @staticmethod
    def subplot(images: Sequence[np.ndarray], titles: Sequence[str], rows: int, cols: int, gray: bool = False, cmap: str = 'gray', sup_title: str = None) -> None: # type: ignore
        """
        Displays multiple images in a subplot with specified layout.

        Parameters:
            images (`Sequence[np.ndarray]`): A list of images to display.
            titles (`Sequence[str]`): A list of titles for the images.
            rows (`int`): Number of rows in the subplot.
            cols (`int`): Number of columns in the subplot.
            gray (`bool`): Whether to display the images in grayscale. Default is `False`.
            cmap (`str`): The color map to use for displaying the images. Default is `'gray'`.
            sup_title (`str`): The super title for the figure. Default is `None`.
        """
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))      # Create a subplot
        for i, (image, title) in enumerate(zip(images, titles)):        # Loop through the images and titles
            row = i // cols     # Calculate the row number
            col = i % cols      # Calculate the column number
            if gray:        # If the image is in grayscale
                axes[row, col].imshow(image, cmap=cmap)     # Display the image in grayscale
            else:       # If the image is in color
                axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))       # Display the image in color
            axes[row, col].set_title(title)     # Set the title of the image
        for ax in axes.flatten()[len(images):]:     # Hide any unused subplots
            ax.axis('off')      # Turn off the axes
        plt.tight_layout()      # Adjust the layout
        if sup_title:       # If a super title is provided
            plt.suptitle(sup_title)     # Set the super title
        plt.show()      # Show the subplot
        
    def adaptiveThresh(self, to_gray: bool = False, channel: int = 0) -> np.ndarray:
        """
        Applies adaptive thresholding to the specified channel of the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the thresholding. Default is `False`.
            channel (`int`): The index of the channel to apply the thresholding on. Default is `0`.

        Returns:
            `np.ndarray`: The thresholded image.

        """
        image = self.gray if to_gray else self.channels[channel]        # Convert the image to grayscale if `to_gray` is True
        blurred_image = self.denoise(image, strategy='gaussian')      # Apply blur to the image to remove noise
        threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)     # Apply adaptive thresholding
        return threshold        # Return the thresholded image
    
    def otsuThresh(self, to_gray: bool = True, channel: int = 0) -> np.ndarray:
        """
        Applies Otsu's thresholding to the specified channel of the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the thresholding. Default is `True`.
            channel (`int`): The index of the channel to apply the thresholding on. Default is `0`.

        Returns:
            `np.ndarray`: The thresholded image.

        """
        if to_gray:     # Convert the image to grayscale if `to_gray` is True
            image = self.gray       # Use the grayscale image
        else:       # If the image is in color
            image = self.channels[channel]      # Use the specified channel
        blurred_image = self.denoise(image, strategy='gaussian')      # Apply blur to the image to remove noise
        _, threshold = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        # Apply Otsu's thresholding
        return threshold        # Return the thresholded image
    
    def denoise(self, image: np.ndarray, strategy: str = 'median') -> np.ndarray:
        """
        Applies denoising to the specified channel of the image.

        Parameters:
            image (`np.ndarray`): The image to denoise.
            strategy (`str`): The denoising strategy to use. Default is `'median'`.

        Returns:
            `np.ndarray`: The denoised image.

        """
        if strategy == 'median':        # If the denoising strategy is median
            denoised_image = cv2.medianBlur(image, 5)       # Apply median blur to the image
        elif strategy == 'gaussian':        # If the denoising strategy is Gaussian
            denoised_image = cv2.GaussianBlur(image, (5, 5), 0)     # Apply Gaussian blur to the image
        else:       # If the denoising strategy is unknown
            raise ValueError("Unknown denoising strategy")      # Raise a ValueError
        return denoised_image       # Return the denoised image

    def KMeans(self, to_gray: bool = True, k: int = 2) -> np.ndarray:
        """
        Applies K-Means clustering to the image.

        Parameters:
            k (`int`): The number of clusters. Default is `2`.

        Returns:
            `np.ndarray`: The clustered image.

        """
        image = self.gray if to_gray else self.image        # Convert the image to grayscale if `to_gray` is True
        Z = image.reshape((-1, 3))      # Reshape the image to a 2D array of pixels with 3 channels
        Z = np.float32(Z)       # Convert the image datatype to float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)       # Define the criteria for the algorithm   
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # Apply K-Means clustering to the image
        centers = np.uint8(centers)         # Convert the centers to 8-bit unsigned integers
        clustered_image = centers[labels.flatten()]     # Get the clustered image
        clustered_image = clustered_image.reshape((image.shape))        # Reshape the clustered image to the original shape
        return clustered_image      # Return the clustered image

    def Snakes(self, to_gray: bool = True) -> np.ndarray:
        """
        Applies active contours (snakes) to the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the active contours. Default is `True`.

        Returns:
            `np.ndarray`: The image with the active contours.

        """
        image = self.gray if to_gray else cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)       # Convert the image to grayscale if `to_gray` is True
        
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)      # Apply Gaussian blur to the image
        edges = cv2.Canny(blurred_image, 50, 150)       # Detect edges in the image using Canny edge detection
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # Find the contours in the image    
        contour_image = np.zeros_like(image)        # Create an empty image to draw the contours
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)       # Draw the contours on the image
        return contour_image        # Return the image with the active contours
# %%
DIR = "/home/sparrow/cv/data/cables"

images = [
    Image("/home/sparrow/cv/data/cables/1.png"),
    Image("/home/sparrow/cv/data/cables/2.png"),
    Image("/home/sparrow/cv/data/cables/3.png"),
    Image("/home/sparrow/cv/data/cables/4.png"),
    Image("/home/sparrow/cv/data/cables/5.png"),
    Image("/home/sparrow/cv/data/cables/6.png"),
    Image("/home/sparrow/cv/data/cables/7.png"),
    Image("/home/sparrow/cv/data/cables/9.png"),
    Image("/home/sparrow/cv/data/cables/11.png"),
    Image("/home/sparrow/cv/data/cables/12.png"),
    Image("/home/sparrow/cv/data/cables/14.png"),
    Image("/home/sparrow/cv/data/cables/15.png"),
    Image("/home/sparrow/cv/data/cables/16.png")
]
#%% 
titles = [Image.filename(image.path) for image in images]
# %%
# Adaptive Thresholding
adaptive_thresholded_images = []

# Apply adaptive thresholding to the images
for i, image in enumerate(images):
    threshold = image.adaptiveThresh(to_gray=True)
    adaptive_thresholded_images.append(threshold)
    
# Denoising
denoised_images = [image.denoise(thresh, strategy='median') for thresh in adaptive_thresholded_images]

# Improve the brightness and contrast of the contour images
contrast_images = [np.log(image + 2) for image in denoised_images]

# Display the images
Image.subplot(contrast_images, titles=["Cable: {}".format(title) for title in titles], rows=3, cols=5, gray=True, cmap='gray', sup_title='Adaptive Thresholding')
# %%
# Otsu Thresholding
otsu_thresholded_images = []

# Apply Otsu's thresholding to the images
for i, image in enumerate(images):
    threshold = image.otsuThresh(to_gray=True)
    otsu_thresholded_images.append(threshold)
    
# Display the images
Image.subplot(otsu_thresholded_images, titles=["Cable: {}".format(title) for title in titles], rows=3, cols=5, gray=True, cmap='gray', sup_title='Otsu Thresholding')
# %%
# Grayscale KMeans Thresholding
kmeans_images = [image.KMeans(k=3) for image in images]

# Display the images
Image.subplot(kmeans_images, titles=["Cable: {}".format(title) for title in titles], rows=3, cols=5, gray=False, cmap='gray', sup_title='KMeans Clustering')
# %%
# Snakes
snake_images = [image.Snakes(to_gray=True) for image in images]

# Display the images
Image.subplot(snake_images, titles=["Cable: {}".format(title) for title in titles], rows=3, cols=5, gray=False, cmap='gray', sup_title='Snakes Thresholding') # type: ignore
# %%
