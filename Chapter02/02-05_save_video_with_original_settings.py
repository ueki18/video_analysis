import cv2

# 入力動画の設定
cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデック指定

# 出力ファイル名、コーデック、FPS、フレームサイズを指定して
# VideoWriterオブジェクトを作成
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 出力動画に書き込む
    out.write(frame)

cap.release()
out.release()
