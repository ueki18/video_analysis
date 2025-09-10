from moviepy import VideoFileClip, concatenate_videoclips, vfx

fade = 2  # フェード時間（秒）

clip1 = VideoFileClip("part1.mp4")
clip2 = VideoFileClip("part2.mp4")

# 映像と必要に応じて音声にも適用されるエフェクトを付与（v2の新方式）
clip1_f = clip1.with_effects([vfx.CrossFadeOut(fade)])
clip2_f = clip2.with_effects([vfx.CrossFadeIn(fade)])

# クリップを重ねるために負のpaddingを指定（解像度差に備えてcompose）
final_clip = concatenate_videoclips([clip1_f, clip2_f],
                                    method="compose",
                                    padding=-fade)

final_clip.write_videofile("output_crossfade.mp4")
