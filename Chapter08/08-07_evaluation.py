import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ダミーデータ生成（4クラス）---
np.random.seed(0)
num_samples = 200
num_classes = 4
feature_dim = 2

X = []
y = []
for cls in range(num_classes):
    X.append(np.random.randn(num_samples // num_classes, feature_dim) + cls * 2)
    y.append(np.full(num_samples // num_classes, cls))

X = np.vstack(X)
y = np.concatenate(y)

# --- 学習・テスト分割 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# --- 分類器（線形SVM）---
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# --- 予測 ---
y_pred = model.predict(X_test)

# --- 評価指標 ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# --- 混同行列のヒートマップ可視化 ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

