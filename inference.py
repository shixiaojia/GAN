import os
import torch
from network import Generator
import cv2
import numpy as np
num_gen = 100

gen = Generator(128)
gen.load_state_dict(torch.load('/home/shixiaojia/dl/gan/ckpt/gen_50.pth'))
gen.eval()

save_path = './generated_pic'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

for i in range(num_gen):
    noisy = torch.randn(1, 128)
    img = gen(noisy)
    img = img.squeeze().permute(1,2,0)
    img = img.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0., 255.).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), bgr)