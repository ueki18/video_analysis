from ultralytics import YOLO

# セグメンテーションモデルをロード（軽量なら n、精度重視なら s/m/l/x）
model = YOLO('yolo11n-seg.pt')

# 画像に対してセグメンテーションを実行
results = model(
    source='input.png',   # 入力画像
    conf=0.3,             # 信頼度のしきい値
    iou=0.5,              # IoUのしきい値
    show=True,            # 結果を画面表示
    save=True             # 結果画像を保存（runs/segment/predict/*）
)
