from deepface import DeepFace

result = DeepFace.verify(
    img1_path = "young_woman.png",
    img2_path = "old_man.png",
    model_name = "Facenet"  # 他に VGG-Face, ArcFace なども選択可
)

print("Is same person:", result["verified"])
print(result)