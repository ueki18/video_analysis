from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

# 動画に対してセグメンテーションを実行
results = model.predict(
    source='input.mp4',         # カメラ入力なら 0
    conf=0.3,
    iou=0.5,
    show=True,                  # 表示（速度重視なら False）
    save=True,                  # マスク付き動画を保存
    project='06-02_YOLO11_seg', # 出力フォルダ
    name='output_dir',          # サブフォルダ名
    stream=True                 # 長尺動画ではメモリ肥大を防ぐため推奨
)
for result in results:
    pass  # 必要に応じて result を処理
