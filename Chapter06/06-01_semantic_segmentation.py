import torch
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# モデルの読み込み
weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet101(weights=weights).eval()

# 入力画像の読み込みと前処理
img = Image.open('input.png')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0)

# セグメンテーション実行
with torch.no_grad():
    output = model(input_tensor)['out'][0]
segmentation = torch.argmax(output, dim=0).byte().cpu().numpy()

# カラーマップ適用
colored = cv2.applyColorMap(segmentation * 10, cv2.COLORMAP_JET)
cv2.imwrite('segmented.png', colored)
