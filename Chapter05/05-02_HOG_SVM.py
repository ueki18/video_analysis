import cv2

# HOG記述子と学習済みSVM（歩行者用）を設定
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('input_full_body.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # マルチスケール検出
    rects, weights = hog.detectMultiScale(
        frame,
        hitThreshold=0,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # 検出結果を描画
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('HOG+SVM Pedestrian', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
