from ultralytics import YOLO

# セグメンテーションモデル（YOLO11n-seg）をロード
model = YOLO("yolo11n-seg.pt")

# 動画ファイルに対してセグメンテーションを実行
results = model(
    source="input.mp4",            # 入力動画ファイル
    conf=0.3,                      # 信頼度のしきい値
    iou=0.5,                       # IoUのしきい値
    show=True,                     # 結果を表示
    save=True,                     # 結果動画を保存
    project="05-03_YOLO11_seg",    # 出力フォルダ
    name="output_dir"              # サブフォルダ名
)
