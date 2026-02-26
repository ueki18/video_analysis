import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# 画像をグレースケールで読み込み
img = cv2.imread('panda.png', cv2.IMREAD_GRAYSCALE)

# HOG特徴量を計算
features, hog_image = hog(img,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          visualize=True)

print("HOG特徴量の次元:", features.shape)

# HOG画像を表示（コントラストを調整）
hog_image_rescaled = exposure.rescale_intensity(hog_image, 
                                                in_range=(0, 10))
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Visualization')
plt.axis('off')
plt.show()