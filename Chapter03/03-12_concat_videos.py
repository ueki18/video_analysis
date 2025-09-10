from moviepy import VideoFileClip, concatenate_videoclips

clip1 = VideoFileClip("part1.mp4")
clip2 = VideoFileClip("part2.mp4")
final_clip = concatenate_videoclips([clip1, clip2])
final_clip.write_videofile("output_merged.mp4")
