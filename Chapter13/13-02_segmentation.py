import cv2
from ultralytics import YOLO

# セグメンテーション用のYOLO26モデルを読み込む
model = YOLO("yolo26n-seg.pt")

img = cv2.imread("input.png")
results = model(img)

# 物体ごとのマスクが描画された画像を取得して保存
result_img = results[0].plot()
cv2.imwrite("seg_result.png", result_img)
