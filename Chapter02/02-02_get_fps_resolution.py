import cv2

cap = cv2.VideoCapture('input.mp4')

# FPSの取得
fps = cap.get(cv2.CAP_PROP_FPS)

# フレームサイズの取得
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, Resolution: {width}x{height}")
