import cv2
import mediapipe as mp

# MediaPipe Pose の初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 映像ファイルの読み込み
video_path = 'soccer.mp4'  # 処理したい動画ファイルを指定
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # 動画の最後に達したら終了
        break

    # RGB変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 姿勢推定結果の描画
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果の表示
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
