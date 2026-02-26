import cv2
import os

cap = cv2.VideoCapture('input.mp4')
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    filename = os.path.join(output_dir, f'frame_{frame_idx:04d}.jpg')
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
