import cv2

cap = cv2.VideoCapture('input.mp4')
ret, frame = cap.read()

# ユーザによるROI選択（初期矩形の指定）
bbox = cv2.selectROI('Select Object', frame, False)
cv2.destroyWindow('Select Object')

# トラッカーを作成（KCF）
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    success, box = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
