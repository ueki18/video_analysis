import cv2

cap = cv2.VideoCapture('input.mp4')
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, curr = cap.read()
    if not ret:
        break

    # 現在のフレームをグレースケールに変換
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # グレースケール画像同士で差分を計算
    diff = cv2.absdiff(prev_gray, curr_gray)
    # 二値化
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('Frame Difference', thresh)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray.copy() # 現在のフレームを次回の比較用に保存

cap.release()
cv2.destroyAllWindows()
