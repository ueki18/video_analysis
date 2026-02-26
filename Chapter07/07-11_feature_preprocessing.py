from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

# CNN特徴（512次元）、Flow特徴（36次元）
cnn_feature = np.random.rand(512)
flow_feature = np.random.rand(36) * 100  # 値のスケールが大きい例

# それぞれを標準化
scaler = StandardScaler()
cnn_norm = scaler.fit_transform(cnn_feature.reshape(-1,1)).flatten()
flow_norm = scaler.fit_transform(flow_feature.reshape(-1,1)).flatten()

# 結合して1本のベクトルに
frame_feature = np.concatenate([cnn_norm, flow_norm])

# L2正規化
frame_feature = normalize(frame_feature.reshape(1, -1))[0]
