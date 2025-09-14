import cv2
import numpy as np

cap = cv2.VideoCapture('input.mp4')
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi法で特徴点を検出
# maxCorners   : 検出する特徴点の最大数
#                （多いほど検出されるが計算量も増える）
# qualityLevel : 最良のコーナーに対する相対品質のしきい値
#                （0.3なら上位30%を採用）
# minDistance  : 特徴点同士の最小距離
#                （小さいと密集した点が選ばれる）
# blockSize    : コーナー検出時に用いる近傍領域のサイズ
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# 描画用マスク画像
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade法でオプティカルフローを計算
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

    # 正しく追跡できた点のみを抽出
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                           (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5,
                               (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('Optical Flow (Lucas-Kanade)', img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        break

cap.release()
cv2.destroyAllWindows()
