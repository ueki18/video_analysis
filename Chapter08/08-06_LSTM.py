import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- ダミーデータ作成 ---
num_samples = 300
sequence_length = 10
feature_dim = 20
num_classes = 3

X = []
y = []
for cls in range(num_classes):
    base = cls * 2
    X_cls = np.random.randn(
        num_samples // num_classes,
        sequence_length,
        feature_dim
    ).astype(np.float32) + base
    y_cls = np.full((num_samples//num_classes,), cls)
    X.append(X_cls)
    y.append(y_cls)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- LSTMモデル ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMClassifier(feature_dim, 64, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 学習 ---
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- 評価と混同行列（数値）---
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = outputs.argmax(dim=1).numpy()
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

print(f"テスト精度: {acc:.4f}")
print("混同行列（行＝正解, 列＝予測）:")
print(cm)
