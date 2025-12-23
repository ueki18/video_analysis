import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('features.csv')  
# 各行が1つの動画サンプルを表し、列には特徴量とラベルが含まれる
X = data.drop('label', axis=1).values
y = data['label'].values

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# SVMによる分類
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, svm_pred))
