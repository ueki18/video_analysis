from PIL import Image
import numpy as np, cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

MODEL_ID = "facebook/mask2former-swin-large-ade-panoptic"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

img = Image.open("input.png").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)
outputs = model(**inputs)

result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[img.size[::-1]]
)[0]
# HxW（ID付き）
panoptic = result["segmentation"].cpu().numpy().astype(np.int32)  

# 可視化（IDごとに色分け）
rand = np.random.default_rng(0)
palette = rand.integers(
    0,
    255,
    size=(panoptic.max() + 1, 3),
    dtype=np.uint8,
)
vis = palette[panoptic]
cv2.imwrite("panoptic_output.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
