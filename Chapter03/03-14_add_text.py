from moviepy import VideoFileClip, TextClip, CompositeVideoClip

clip = VideoFileClip("input.mp4")

# 例：フォントは環境のTTFパスを指定（Windowsの例）
font_path = "C:/Windows/Fonts/arial.ttf"

txt = TextClip(
    text="Sample Title",
    method="label",
    font_size=50,
    color="white",
    font=font_path
).with_position("center").with_duration(clip.duration)

video = CompositeVideoClip([clip, txt])
video.write_videofile("output_with_text.mp4")
