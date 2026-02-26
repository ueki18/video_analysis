import cv2

def sample_frames(video_path, interval=30, max_frames=30):
    """
    動画から一定間隔でフレームをサンプリングし、固定長のリストを返す。

    Args:
        video_path (str): 動画ファイルのパス
        interval (int): 何フレームごとにサンプリングするか
        max_frames (int): 出力する最大フレーム数（不足時はパディング）

    Returns:
        list: サンプリングしたフレーム画像のリスト
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        count += 1

    cap.release()

    # 足りない場合は最後のフレームでパディング
    while len(frames) < max_frames:
        frames.append(frames[-1])

    print(f"抽出されたフレーム数: {len(frames)}")
    return frames

# 使用例
frames = sample_frames("input.mp4", interval=15, max_frames=30)
