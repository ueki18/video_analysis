from ultralytics import YOLO
import cv2

# 軽量モデル例：yolo11n-pose.pt（他に yolo11s-pose.pt なども使用可）
model = YOLO("yolo11n-pose.pt")

# --- 動画ファイルの読み込み ---
video_path = "soccer.mp4"  # 処理したい動画のパスを指定
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # 動画の最後でループ終了
        break

    # 骨格推定を含む推論
    results = model(frame)

    # 描画済みフレームを取得（bounding box + keypoints）
    annotated = results[0].plot()

    # 表示
    cv2.imshow("YOLO v11 Pose Estimation", annotated)

    # 'q'キーで途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
