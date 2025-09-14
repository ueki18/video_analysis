import cv2

cap = cv2.VideoCapture('input.avi')

# MOG2背景差分器を作成（引数は適宜調整）
# history: 履歴として保持するフレーム数。
#          大きいほど安定するが、メモリ使用量が増える。
# varThreshold: 背景とみなすためのしきい値。
#               小さいほど変化に敏感になる。
# detectShadows: 影を検出して前景から除外するかどうか。
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=16, 
    detectShadows=True
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)  # 前景領域を抽出
    cv2.imshow('MOG2 Foreground', fgmask)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
