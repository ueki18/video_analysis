import subprocess

cmd = [
    "ffmpeg",
    "-i", "input.mp4",      # 入力動画
    "-vf", "fps=1",         # 1秒ごとにフレームを抽出
    "frame_%03d.jpg"        # 出力ファイル名（連番付き）
]
subprocess.run(cmd)
