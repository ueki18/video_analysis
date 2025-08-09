from deepface import DeepFace

analysis = DeepFace.analyze(
    img_path="old_man.png",
    actions=["age", "gender", "emotion"]
)

print("年齢:", analysis[0]["age"])
print("性別:", analysis[0]["dominant_gender"])
print("感情:", analysis[0]["dominant_emotion"])
