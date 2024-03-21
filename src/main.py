import cv2
import matplotlib.pyplot as plt
from functions.imgproc import gamma_correction, median_filtering

plt.style.use("seaborn-v0_8-paper")

# Reading the image/s
Path: str = "/home/sparrow/cv/images"
img_gray = cv2.imread(Path + "/gray9.jpg", cv2.IMREAD_GRAYSCALE)
img_dark = cv2.imread(Path + "/dark9.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Dark Image", img_dark)
cv2.waitKey(5000)

img_dark_corrected = gamma_correction(img_dark, 0.2)
img_dark_filtered = median_filtering(img_dark, 7)

cv2.imshow("Median Filtered Image (kernel = 3)", img_dark_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(
    img_dark_filtered.ravel(),
    256,
    [0, 256],
    label = "Filtered Dark Image",
);
plt.title("Median Filtered Image (kernel = 3)");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();