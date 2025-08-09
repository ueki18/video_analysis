import cv2

cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# コーデックと出力設定（カラー=False でモノクロ動画として保存）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_gray.mp4', fourcc, fps, (width, height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(gray)

cap.release()
out.release()
