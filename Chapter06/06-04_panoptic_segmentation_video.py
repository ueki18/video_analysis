import cv2
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

MODEL_ID = "facebook/mask2former-swin-large-ade-panoptic"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cap = cv2.VideoCapture("input.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

palette = None
while True:
    ok, frame = cap.read()
    if not ok:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]

    seg = result["segmentation"].cpu().numpy().astype(np.int32)
    if palette is None:
        rng = np.random.default_rng(0)
        palette = rng.integers(0, 255, size=(seg.max() + 1, 3), dtype=np.uint8)
    vis = palette[seg]

    # 表示（RGB→BGRに変換してOpenCV表示）
    cv2.imshow("Panoptic Segmentation", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 'q'キーで終了
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
