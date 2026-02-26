import cv2
import os

cap = cv2.VideoCapture('input.mp4')
output_dir = 'skip_frames'
os.makedirs(output_dir, exist_ok=True)

frame_idx = 0
save_interval = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % save_interval == 0:
        filename = os.path.join(output_dir, 
                                f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
