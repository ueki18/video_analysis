import cv2
import numpy as np

# 連続する2フレームをグレースケールで読み込み
prev = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
next = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)

# オプティカルフローを計算
flow = cv2.calcOpticalFlowFarneback(prev, next,
                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)

# 大きさと方向に変換
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# 平均の動きの大きさと方向ヒストグラムを特徴ベクトルとする
mean_mag = np.mean(mag)
hist, _ = np.histogram(ang, bins=8, range=(0, 2*np.pi), weights=mag)
features = np.concatenate(([mean_mag], hist))

print("オプティカルフロー特徴量:", features)
