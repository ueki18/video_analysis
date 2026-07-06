import cv2
from ultralytics import YOLO

# YOLO26のモデルを読み込む（nは最も軽量なモデル）
model = YOLO("yolo26n.pt")

# 画像を読み込んで物体検出を実行
img = cv2.imread("input.png")
results = model(img)

# 検出結果（バウンディングボックス等）が描画された画像を取得して保存
result_img = results[0].plot()
cv2.imwrite("yolo_result.png", result_img)

# 検出された各物体のクラス名・信頼度を表示
for box in results[0].boxes:
    class_name = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"{class_name}: {confidence:.2f}")
