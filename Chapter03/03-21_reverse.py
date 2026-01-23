from moviepy import VideoFileClip, vfx

clip = VideoFileClip("input.mp4")
edited = clip.with_effects([vfx.TimeMirror()])
edited.write_videofile("output_reverse.mp4")
