from moviepy import VideoFileClip, vfx

clip = VideoFileClip("input.mp4")
edited = clip.with_effects([vfx.MirrorX()])
edited.write_videofile("output_mirrorx.mp4")
