import torch
B, T, D = 32, 30, 526
x = torch.randn(B, T, D)        # (B,T,D)

# LSTM へ: batch_first=True を推奨
lstm = torch.nn.LSTM(input_size=D, hidden_size=256, batch_first=True)
y, _ = lstm(x)                  # y: (B,T,256)

# Transformer へ: attention mask（パディング部=0）を付与
# 例: すべて実フレームなら全て1
mask = torch.ones(B, T, dtype=torch.bool)  
encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=D,
    nhead=2,
    batch_first=True
)
enc = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
z = enc(x, src_key_padding_mask=~mask)     # z: (B,T,D)
