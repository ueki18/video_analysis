from ultralytics import YOLO
import cv2

# 軽量モデル例：yolo11n‑pose.pt 他に yolov11s‑pose.pt など
model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # 骨格推定を含む推論

    # 描画済みフレームを取得（bounding box + keypoints）
    annotated = results[0].plot()

    cv2.imshow("YOLO v11 Pose Estimation", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
