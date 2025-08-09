from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py'
checkpoint_file = 'https://download.openmmlab.com/mmaction/slowfast/slowfast_r50_kinetics400.pth'

model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
results = inference_recognizer(model, 'squat.mp4')
print(results)
