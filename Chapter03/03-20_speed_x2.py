from moviepy import VideoFileClip

clip = VideoFileClip("input.mp4")
edited = clip.with_speed_scaled(factor=2) # 2倍速
edited.write_videofile("output_speed_x2.mp4")
