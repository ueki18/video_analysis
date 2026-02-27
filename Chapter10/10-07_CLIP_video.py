import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import open_clip

# ===== 設定 =====
# 解析したい動画ファイル
VIDEO_PATH = "multiple_scene_changes.mp4"
# 例：カップルがレストランで食事
QUERY_TEXT = "a couple dining at a restaurant"
# 何秒ごとにフレームをサンプルするか
INTERVAL_SEC = 0.1
# フレームのバッチ処理サイズ
BATCH_SIZE = 32

# ===== OpenCLIP 準備 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

# テキスト埋め込み
with torch.no_grad():
    text_tokens = tokenizer([QUERY_TEXT]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )

# ===== 動画読み込み =====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
stride = max(int(round(fps * INTERVAL_SEC)), 1)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

timestamps = []
preprocessed_frames = []

frame_idx = 0
sample_idx = 0

# ===== フレームサンプリング =====
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % stride == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = preprocess(pil_img)
        preprocessed_frames.append(tensor)
        timestamps.append(frame_idx / fps)
        sample_idx += 1
    frame_idx += 1

cap.release()

if len(preprocessed_frames) == 0:
    raise RuntimeError("No frames sampled. Try decreasing INTERVAL_SEC.")

# ===== 画像埋め込み（バッチ処理） =====
image_features_all = []
with torch.no_grad():
    for i in range(0, len(preprocessed_frames), BATCH_SIZE):
        batch = torch.stack(
            preprocessed_frames[i:i + BATCH_SIZE]
        ).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        image_features_all.append(feats)

image_features = torch.cat(image_features_all, dim=0)

# ===== 類似度計算 =====
scores = (image_features @ text_features.T).squeeze(1)
scores = scores.detach().cpu().numpy()

# ===== 結果表示 =====
best_i = int(np.argmax(scores))
print(
    f"Best match at {timestamps[best_i]:.2f}s, "
    f"score={scores[best_i]:.4f}"
)

plt.figure()
plt.plot(timestamps, scores)
plt.xlabel("Time (s)")
plt.ylabel("Similarity")
plt.title(f'Similarity over time: "{QUERY_TEXT}"')
plt.grid(True)
plt.tight_layout()
plt.show()
