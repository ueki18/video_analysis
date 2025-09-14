from ultralytics import YOLO

# YOLO12n モデルをロード
model = YOLO('yolo12n.pt')

# 動画ファイルに対して物体検出のみを実行（保存と表示も可能）
results = model(
    source="input.mp4",            # 入力動画ファイル
    conf=0.3,                      # 信頼度のしきい値
    iou=0.5,                       # IoUのしきい値
    show=True,                     # 検出結果を画面に表示
    save=True,                     # 検出結果付き動画を保存
    project='05-02_YOLO12_video',  # 出力フォルダ
    name='output_dir'              # サブフォルダ名
)
