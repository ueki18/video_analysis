import cv2

cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# コーデックと出力設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_gaussian_blur.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    out.write(blurred)

cap.release()
out.release()
