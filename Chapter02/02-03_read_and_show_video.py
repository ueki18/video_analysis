import cv2

# 動画ファイルの読み込み
cap = cv2.VideoCapture('input.mp4')

while cap.isOpened():
    # フレームを1つずつ読み込む
    ret, frame = cap.read()

    # 読み込めなければ終了
    if not ret:
        break

    # フレームを表示
    cv2.imshow('Frame', frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()

# 開いているすべてのウィンドウを閉じる
cv2.destroyAllWindows()
