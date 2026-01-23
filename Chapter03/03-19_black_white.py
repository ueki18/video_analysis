from moviepy import VideoFileClip, vfx

clip = VideoFileClip("input.mp4")
edited = clip.with_effects([vfx.BlackAndWhite()])
edited.write_videofile("output_bw.mp4")
