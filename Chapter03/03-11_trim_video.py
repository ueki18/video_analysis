from moviepy import VideoFileClip

clip = VideoFileClip("input.mp4")
subclip = clip.subclipped(2, 4)  # 2秒〜4秒
subclip.write_videofile("output_trimmed.mp4")
