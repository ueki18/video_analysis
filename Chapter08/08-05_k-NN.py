import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# k-NNによる分類
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
print('k-NN Accuracy:', accuracy_score(y_test, knn_pred))
