from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 一時回避（長期運用は非推奨）

model = YOLO('yolo12n.pt')

# 結果を逐次処理することで長尺動画でもメモリ肥大を防ぐ
for r in model.track(
    source='input.mp4',
    show=True,            # 速度重視なら False に
    save=True,
    tracker='botsort.yaml',
    stream=True
):
    pass
