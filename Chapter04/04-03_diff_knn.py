import cv2

cap = cv2.VideoCapture('input.avi')

# KNN背景差分器を作成（引数は適宜調整）
# history: 背景モデルの履歴フレーム数。
#          大きいほど安定するが、環境変化に追従しにくくなる。
# dist2Threshold: 背景とみなす距離の二乗のしきい値。
#                 小さいほど変化に敏感になる。
# detectShadows: 影を検出して前景から除外するかどうか。
fgbg = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('KNN Foreground', fgmask)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
