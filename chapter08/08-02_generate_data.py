import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# 人工データ生成（make_classification を使用）
X, y = make_classification(
    n_samples=5000,    # サンプル数
    n_features=50,     # 特徴量の次元数
    n_classes=3,       # クラス数
    n_informative=30,  # 実際に分類に寄与する特徴量（有効特徴量）
    n_redundant=10,    # 他の特徴量の線形結合で表せる冗長特徴量
    class_sep=2.0,     # クラス間の分離度（大きいほど分類が容易）
    flip_y=0.02,       # ラベルをランダムに入れ替える割合（ノイズ率）
    random_state=42    # 乱数シード（再現性のため固定）
)

# DataFrame に変換して保存
cols = [f"f{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df["label"] = y
df.to_csv("features.csv", index=False)
