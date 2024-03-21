import cv2
import matplotlib.pyplot as plt
from functions.imgproc import gamma_correction

plt.style.use("seaborn-v0_8-paper")

# Reading the image/s
Path: str = "/home/sparrow/cv/images"
img_gray = cv2.imread(Path + "/gray9.jpg", cv2.IMREAD_GRAYSCALE)
img_dark = cv2.imread(Path + "/dark9.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Dark Image", img_dark)
cv2.waitKey(5000)

img_dark_corrected = gamma_correction(img_dark, 0.2)
cv2.imshow("Gamma Corrected Image (gamma = 0.2)", img_dark_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(
    img_dark_corrected.ravel(),
    256,
    [0, 256],
    label = "Corrected Dark Image",
);
plt.title("Gamma Corrected Image (Gamma = 0.2)");
plt.xlabel("Gray Level");
plt.ylabel("Count");
plt.show();