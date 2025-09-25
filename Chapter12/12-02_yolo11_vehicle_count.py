import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 応急処置
os.environ["OMP_NUM_THREADS"] = "1"          # スレッド数を抑える(任意)

from ultralytics import YOLO
import cv2

# 1) モデル読み込み
model = YOLO("yolo11n.pt")  # 軽量
allowed = {"car", "bus", "truck", "motorcycle"}  # 数えたいクラスだけ

cap = cv2.VideoCapture("british_highway_traffic.mp4")
line_y = 300
count = 0

# 状態保持
counted = set()          # 交差済みID
last_center_y = {}       # 前フレームの中心y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2) トラッキングを有効化（安定IDを得る）
    #    conf/iouは適宜調整
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", 
        conf=0.4, iou=0.5
    )

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Traffic Count", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes
    names = model.names  # クラス名辞書

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        cls_name = names[cls_id]
        if cls_name not in allowed:
            continue  # 3) 人物などは無視

        # 安定ID（トラッカが付与）。ないならスキップ
        if boxes.id is None:
            continue
        obj_id = int(boxes.id[i])

        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 4) 上→下の交差イベントでだけカウント
        prev = last_center_y.get(obj_id)
        if prev is not None:
            crossed_down = (prev <= line_y) and (cy > line_y)
            if crossed_down and obj_id not in counted:
                count += 1
                counted.add(obj_id)

        last_center_y[obj_id] = cy

        # 可視化
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} #{obj_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ラインとカウント表示
    cv2.line(frame, (0, line_y), 
             (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow("Traffic Count", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
