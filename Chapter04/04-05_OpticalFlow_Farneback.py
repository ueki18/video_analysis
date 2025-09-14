import cv2
import numpy as np

cap = cv2.VideoCapture('input.mp4')
ret, prev = cap.read()
if not ret:
    raise RuntimeError("最初のフレームを読み込めませんでした")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# HSV画像を用意（彩度を最大に設定）
hsv = np.zeros_like(prev)
hsv[..., 1] = 255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Farnebäck法でオプティカルフローを計算
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5,    # ピラミッドスケール（各レベルで画像を縮小する比率）
        levels=3,         # ピラミッドレベル数（多いほど大きな動きに対応）
        winsize=15,       # ウィンドウサイズ（大きいほど滑らか、小さいほど細部に敏感）
        iterations=3,     # 各レベルでの反復回数
        poly_n=5,         # 多項式近似の近傍サイズ
        poly_sigma=1.2,   # 多項式近似に用いるガウシアンのσ
        flags=0           # オプションフラグ（通常は0）
    )

    # 動きベクトルを極座標に変換
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 方向を色相に，動きの大きさを明度に変換
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # HSV→BGRに変換して可視化
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow (Farneback)', rgb)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray

cap.release()
cv2.destroyAllWindows()
