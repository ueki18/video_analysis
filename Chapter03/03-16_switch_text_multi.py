from moviepy import VideoFileClip, TextClip, CompositeVideoClip

clip = VideoFileClip("input.mp4")
font_path = "C:/Windows/Fonts/arial.ttf"

# (表示文字, 開始秒, 終了秒, 色) のタプルで定義
schedule = [
    ("Intro", 0, 2, "white"),
    ("Scene A", 2, 4, "yellow"),
    ("Scene B", 4, 6, "cyan"),
]

text_clips = [
    TextClip(
        text=msg,
        method="label",
        font_size=50,
        color=color,
        font=font_path
    ).with_position("center").with_start(start).with_duration(end - start)
    for (msg, start, end, color) in schedule
]

video = CompositeVideoClip([clip, *text_clips])
video.write_videofile("output_switch_text_multi.mp4")
