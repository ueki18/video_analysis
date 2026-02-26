import numpy as np

# 各特徴ベクトル（例）
cnn_feature = np.random.rand(512)      # CNN埋め込み
geom_feature = np.array([cx, cy, w, h, aspect_ratio])  # 幾何特徴
flow_feature = np.random.rand(9)       # オプティカルフローヒストグラム

# Concatenation による統合
frame_feature = np.concatenate([
    cnn_feature,
    geom_feature,
    flow_feature
])

print("統合後の特徴ベクトル次元:", frame_feature.shape)
