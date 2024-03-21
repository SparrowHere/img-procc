import cv2
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

# Reading the image/s
Path: str = "/home/sparrow/cv/images"
img_gray = cv2.imread(Path + "/gray9.jpg", cv2.IMREAD_GRAYSCALE)
img_dark = cv2.imread(Path + "/dark9.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Dark Image", img_dark)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Getting histogram of the image
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