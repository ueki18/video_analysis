import numpy as np

# ----- 1) 正常データ(ダミー)の用意: 2次元ガウス分布に従うNサンプル -----
rng = np.random.default_rng(42)
mu_true   = np.array([0.0, 0.0])  # 真の平均(ダミー)
Sigma_true= np.array([[1.0, 0.5],
                      [0.5, 1.0]])  # 真の共分散(ダミー)
N = 200
normal_data = rng.multivariate_normal(mu_true, Sigma_true, size=N)

# ----- 2) 正常モデルの推定: 平均ベクトルmu, 共分散行列Sigma -----
mu_est    = normal_data.mean(axis=0)
Sigma_est = np.cov(normal_data, rowvar=False)             # 形状(2,2)
Sigma_inv = np.linalg.inv(Sigma_est)                      # 逆行列

def mahalanobis_sq(x, mu, Sigma_inv):
    """
    (x - mu)^T Sigma_inv (x - mu) を返す．
    （平方距離 = 外れ度スコア）
    """
    d = x - mu
    return float(d.T @ Sigma_inv @ d)

# ----- 3) 評価したいサンプル(ダミー) -----
candidates = np.array([
    [ 0.2, -0.1],  # 典型的に正常そう
    [ 1.0,  1.0],  # 少し外れ気味
    [ 3.0,  3.0],  # 強い外れ
    [-2.5,  2.8],  # 強い外れ
])

# ----- 4) 外れ度(異常度)を算出し, しきい値で判定 -----
# 2次元のカイ二乗分布の95%点はおよそ 5.991
CHI2_95 = 5.991

print("推定平均 mu =", mu_est)
print("推定共分散 Sigma =\n", Sigma_est)
print()

for i, x in enumerate(candidates):
    score = mahalanobis_sq(x, mu_est, Sigma_inv)  # 外れ度(大きいほど異常)
    is_anom = score > CHI2_95
    print(f"sample {i}: x={x},  score(Mahalanobis^2)={score:.3f}  ->  "
          f"{'ANOMALY' if is_anom else 'normal'}")
