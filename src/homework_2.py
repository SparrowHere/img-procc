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
    def __init__(self, path: str) -> None:
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        self.gray: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.channels: Sequence[np.ndarray] = cv2.split(self.image)
        self.store_evolution_in = lambda lst: lambda x: lst.append(np.copy(x))
    
    @staticmethod
    def filename(filename: str) -> int:
        """
        Extracts the number from the filename.

        Parameters:
            filename (`str`): The filename from which to extract the number.

        Returns:
            `int`: The extracted number.

        """
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            return -1
        
    def show(self, image: np.ndarray = None, gray: bool = False, title: str = 'Image') -> None: # type: ignore
        """
        Displays the image.

        Parameters:
            image (`np.ndarray`): The image to display. If None, display the original image.
            gray (`bool`): Whether to display the image in grayscale. Default is `False`.
            title (`str`): The title of the window. Default is `'Image'`.

        """
        if image is None:
            image = self.gray if gray else self.image
        if gray:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

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
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        for i, (image, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            if gray:
                axes[row, col].imshow(image, cmap=cmap)
            else:
                axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title)
        for ax in axes.flatten()[len(images):]:  # Hide any unused subplots
            ax.axis('off')
        plt.tight_layout()
        if sup_title:
            plt.suptitle(sup_title)
        plt.show()
        
    def adaptiveThresh(self, to_gray: bool = False, channel: int = 0) -> np.ndarray:
        """
        Applies adaptive thresholding to the specified channel of the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the thresholding. Default is `False`.
            channel (`int`): The index of the channel to apply the thresholding on. Default is `0`.

        Returns:
            `np.ndarray`: The thresholded image.

        """
        image = self.gray if to_gray else self.channels[channel]
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)      # Apply blur to the image to remove noise
        threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        return threshold
    
    def otsuThresh(self, to_gray: bool = True, channel: int = 0) -> np.ndarray:
        """
        Applies Otsu's thresholding to the specified channel of the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the thresholding. Default is `True`.
            channel (`int`): The index of the channel to apply the thresholding on. Default is `0`.

        Returns:
            `np.ndarray`: The thresholded image.

        """
        if to_gray:
            image = self.gray
        else:
            image = self.channels[channel]
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)      # Apply blur to the image to remove noise  
        _, threshold = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold
    
    def denoise(self, image: np.ndarray, strategy: str = 'median') -> np.ndarray:
        """
        Applies denoising to the specified channel of the image.

        Parameters:
            image (`np.ndarray`): The image to denoise.
            strategy (`str`): The denoising strategy to use. Default is `'median'`.

        Returns:
            `np.ndarray`: The denoised image.

        """
        if strategy == 'median':
            denoised_image = cv2.medianBlur(image, 5)
        elif strategy == 'gaussian':
            denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
        else:
            raise ValueError("Unknown denoising strategy")
        return denoised_image

    def KMeans(self, to_gray: bool = True, k: int = 2) -> np.ndarray:
        """
        Applies K-Means clustering to the image.

        Parameters:
            k (`int`): The number of clusters. Default is `2`.

        Returns:
            `np.ndarray`: The clustered image.

        """
        image = self.gray if to_gray else self.image
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore
        centers = np.uint8(centers)
        clustered_image = centers[labels.flatten()] # type: ignore
        clustered_image = clustered_image.reshape((image.shape))
        return clustered_image

    def Snakes(self, to_gray: bool = True) -> np.ndarray:
        """
        Applies active contours (snakes) to the image.

        Parameters:
            to_gray (`bool`): Whether to convert the image to grayscale before applying the active contours. Default is `True`.

        Returns:
            `np.ndarray`: The image with the active contours.

        """
        image = self.gray if to_gray else cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        return contour_image
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
