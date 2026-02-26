import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 特徴量とラベルの読み込み
data = pd.read_csv('features.csv')
# 各行が1つの動画サンプルを表し、列には特徴量とラベルが含まれる
X = data.drop('label', axis=1).values
y = data['label'].values

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ランダムフォレストによる分類
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))

# 特徴量の重要度分析
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.title('Feature Importances')
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), indices[:10])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
