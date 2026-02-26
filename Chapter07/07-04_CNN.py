import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torchvision.models import ResNet18_Weights

# ResNet18モデルをロード（ImageNetで事前学習済み）
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 最後の全結合層（分類層）を除去して特徴抽出器とする
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# 前処理：ResNetの入力に合わせてリサイズ・正規化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# フレーム画像を読み込み（OpenCV形式）
img = cv2.imread('panda.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img)

# テンソル変換
tensor = transform(pil_img).unsqueeze(0)

# 特徴抽出
with torch.no_grad():
    feature = model(tensor)

print("特徴ベクトルの形状:", feature.shape)
