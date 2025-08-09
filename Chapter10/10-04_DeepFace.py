import cv2
from deepface import DeepFace

# === Settings ===
reference_image = "target.jpg"       # 照合対象の画像（静止画）
video_path = "input_video.mp4"       # 入力映像（動画）
frame_interval = 5                   # 照合間隔（フレーム数）

# === Load video ===
cap = cv2.VideoCapture(video_path)
frame_id = 0
result_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        # Save current frame to a temporary image
        cv2.imwrite("temp.jpg", frame)

        try:
            result = DeepFace.verify(
                img1_path=reference_image,
                img2_path="temp.jpg",
                model_name="Facenet",
                enforce_detection=False  # 顔がなくてもエラーを出さない
            )

            if result["verified"]:
                result_text = "Match"
            else:
                result_text = "No Match"

        except Exception as e:
            result_text = "No Face Found"

    # Draw result text on the frame
    display_frame = frame.copy()
    cv2.putText(display_frame,
                f"Frame {frame_id}: {result_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2)

    # Show the frame
    cv2.imshow("Verification", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
