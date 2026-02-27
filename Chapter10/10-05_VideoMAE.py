from transformers import AutoProcessor, AutoModelForVideoClassification
import torch
import av
import numpy as np

VIDEO_PATH = "ice_skating.mp4"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- 動画の読み込みとサンプリング ---
container = av.open(VIDEO_PATH)
frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
container.close()

num_frames = getattr(model.config, "num_frames", 16)
idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
video = [frames[i] for i in idx]

# --- 前処理と推論 ---
inputs = processor(video, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# --- softmaxでスコア化 ---
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
topk = torch.topk(probs, k=5)  # 上位5件を表示

print("Top predictions:")
for score, idx in zip(topk.values, topk.indices):
    label = model.config.id2label[idx.item()]
    print(f"{label:30s}  score={score.item():.4f}")

