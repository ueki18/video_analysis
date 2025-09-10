from moviepy import VideoFileClip, TextClip, CompositeVideoClip

clip = VideoFileClip("input.mp4")
font_path = "C:/Windows/Fonts/arial.ttf"  # 利用可能なフォントのパス

# 0〜3秒に表示するテキスト
txt1 = TextClip(
    text="First Title",
    method="label",
    font_size=50,
    color="white",
    font=font_path
).with_position("center").with_duration(3)

# 3〜6秒に表示するテキスト
txt2 = TextClip(
    text="Second Title",
    method="label",
    font_size=50,
    color="yellow",
    font=font_path
).with_position("center").with_start(3).with_duration(3)

video = CompositeVideoClip([clip, txt1, txt2])
video.write_videofile("output_switch_text.mp4")