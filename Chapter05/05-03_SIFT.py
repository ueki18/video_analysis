import cv2

# --- 基準画像を読み込み、特徴点抽出 ---
img_query = cv2.imread('query.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp_query, des_query = sift.detectAndCompute(img_query, None)

# --- 特徴点マッチング器（BFMatcher） ---
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# --- 動画読み込み ---
cap = cv2.VideoCapture('input_for_SIFT.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # フレームから特徴点抽出
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    if des_frame is not None and len(kp_frame) >= 2:
        # マッチング実行
        matches = bf.match(des_query, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # 上位マッチを描画（例：前20件）
        img_match = cv2.drawMatches(
            img_query, kp_query, frame, kp_frame, matches[:20], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow('Matching', img_match)
    else:
        cv2.imshow('Matching', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
