import cv2

# 2つの連続フレームを読み込み（例: frame1, frame2）
frame1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)

# 差分画像を計算
diff = cv2.absdiff(frame1, frame2)

# しきい値処理で動きの領域を抽出
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# ファイルとして保存
cv2.imwrite("frame_diff_result.png", thresh)
