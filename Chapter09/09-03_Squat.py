import cv2
import mediapipe as mp

video_path = 'squat.mp4'
cap = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawer = mp.solutions.drawing_utils

count = 0
state = 'up'  # 最初は立っている状態

threshold_down = 0.08  # しゃがみ判定（diffが小さいとしゃがみ）
threshold_up = 0.12    # 立ち上がり判定（diffが大きいと立ち）

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        hips = [
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.RIGHT_HIP],
        ]
        knees = [
            lm[mp_pose.PoseLandmark.LEFT_KNEE],
            lm[mp_pose.PoseLandmark.RIGHT_KNEE],
        ]

        hip = max(hips, key=lambda p: p.visibility)
        knee = max(knees, key=lambda p: p.visibility)

        # MediaPipeのy座標は下向きが正（上が0、下が1）
        # 直立時は差が大きく、しゃがむと差が小さくなる
        diff = knee.y - hip.y  # 膝の高さ − 腰の高さ

        if hip.visibility > 0.5 and knee.visibility > 0.5:
            # しゃがみ判定（腰が下がって膝とのy座標差が縮まったとき）
            if state == 'up' and diff < threshold_down:
                state = 'down'

            # 立ち上がり判定（腰が上がって膝とのy座標差が広がったとき）
            elif state == 'down' and diff > threshold_up:
                count += 1
                state = 'up'

        drawer.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        # 表示
        cv2.putText(frame, f"hip.y: {hip.y:.3f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"knee.y: {knee.y:.3f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"diff: {diff:.3f}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"State: {state.upper()}", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if state == 'up' else (0, 128, 255), 2)

    cv2.putText(frame, f"Squats: {count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('Squat Detection (corrected)', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
