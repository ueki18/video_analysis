import cv2

cap = cv2.VideoCapture('input.mp4')
ret, prev = cap.read()

while cap.isOpened():
    ret, curr = cap.read()
    if not ret:
        break

    # フレーム間の絶対差分
    diff = cv2.absdiff(prev, curr)
    # グレースケールに変換
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 二値化
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    cv2.imshow('Frame Difference', thresh)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    prev = curr  # 現在のフレームを次回の比較用に保存

cap.release()
cv2.destroyAllWindows()
