from ultralytics import YOLO
import cv2

# 事前学習済みYOLOモデルをロード
model = YOLO('yolo11s.pt')

# 画像を読み込み
img = cv2.imread('woman.png')

# 推論を実行
results = model(img)[0]

# 人物クラスのみ処理
for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
    if int(cls) == 0:  # 'person' クラス
        x_min, y_min, x_max, y_max = map(int, box)
        w, h = x_max - x_min, y_max - y_min
        cx, cy = x_min + w / 2, y_min + h / 2
        aspect_ratio = w / h

        print(f"Aspect Ratio: {aspect_ratio:.2f}, "
              f"Centroid: ({cx:.1f}, {cy:.1f}), "
              f"Width: {w}, Height: {h}")

        # 検出結果を画像に描画
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# 結果を保存
cv2.imwrite('woman_result.png', img)
