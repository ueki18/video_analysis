import cv2

# デバイスID 0 のカメラを指定
cap = cv2.VideoCapture(0)

# 接続確認
if not cap.isOpened():
    print("Webカメラにアクセスできません。")
    exit()

while True:
    # フレームを取得
    ret, frame = cap.read()

    # フレームが取得できない場合は終了
    if not ret:
        break

    # フレームを表示
    cv2.imshow("Webcam", frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()
