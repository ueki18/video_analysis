import cv2

cap = cv2.VideoCapture('input.mp4')
ret, frame = cap.read()

# 複数のROIを選択
bboxes = cv2.selectROIs('Select Objects', frame, False)
cv2.destroyWindow('Select Objects')

# MultiTrackerの初期化（例：CSRTを使用）
multi_tracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    tracker = cv2.legacy.TrackerCSRT_create()
    multi_tracker.add(tracker, frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    success, boxes = multi_tracker.update(frame)
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Multi Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
