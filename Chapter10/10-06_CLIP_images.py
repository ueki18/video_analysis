from pathlib import Path
from PIL import Image
import torch, open_clip

# ---- 設定 ----
IMAGES_DIR = Path("animals_20")  # 画像ディレクトリ
QUERY_ANIMAL = "panda"           # 検索したい動物名
TOPK = 10

# ---- モデル読み込み ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

# ---- 画像の読み込み ----
exts = {".jpg", ".jpeg", ".png", ".bmp"}
image_paths = sorted([
    p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts
])

images = [
    preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
    for p in image_paths
]
image_input = torch.cat(images).to(device)

# ---- 特徴抽出 ----
with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# ---- テキスト特徴 ----
text = [f"a photo of {QUERY_ANIMAL}"]
with torch.no_grad():
    text_tokens = tokenizer(text).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ---- 類似度計算 ----
scores = (image_features @ text_features.T).squeeze(1)
topk = min(TOPK, len(image_paths))
rank_idx = torch.topk(scores, k=topk).indices.tolist()

# ---- 結果表示 ----
print(f"\nQuery: {QUERY_ANIMAL}")
print(f"Top-{topk} results:\n")
for rank, i in enumerate(rank_idx, 1):
    print(f"{rank:2d}. {image_paths[i].name:<30s}  "
          f"score={scores[i].item():.4f}")
