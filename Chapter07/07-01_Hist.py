import cv2
import matplotlib.pyplot as plt

# 画像をグレースケールで読み込み
img = cv2.imread('panda.png', cv2.IMREAD_GRAYSCALE)

# 輝度ヒストグラム（16ビン）を計算
hist = cv2.calcHist([img], [0], None, [16], [0, 256])

# ヒストグラムの値を表示
print(hist.flatten())

# ヒストグラムをグラフで表示
plt.bar(range(16), hist.flatten(), width=1.0, edgecolor='black')
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Grayscale Histogram (16 bins)')
plt.show()
