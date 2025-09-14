import cv2

# 顔検出用のHaar Cascade分類器を読み込み（OpenCV付属の学習済みデータ）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('input.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検出（返り値は矩形領域のリスト）
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 検出した顔に矩形を描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
