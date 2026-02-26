import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# 画像をグレースケールで読み込み
img = cv2.imread('panda.png', cv2.IMREAD_GRAYSCALE)

# LBP特徴量を計算（近傍点8, 半径1）
lbp = local_binary_pattern(img, P=8, R=1, method='uniform')

# LBPの値を一部表示
print(lbp[:5, :5])  # 左上5x5領域の値を表示

# LBP画像を表示
plt.imshow(lbp, cmap='gray')
plt.title('LBP Image')
plt.axis('off')
plt.show()
