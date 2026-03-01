from moviepy import VideoFileClip, vfx

clip = VideoFileClip("input.mp4")
edited = clip.with_effects([
    vfx.Resize(height=360),  # 高さ360にリサイズ
    vfx.Rotate(180)          # 180度回転（上下反転）
])

edited.write_videofile("output_rotated.mp4")

