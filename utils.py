import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import glob


class MyDataset(Dataset):
    def __init__(self, img_path, device):
        super(MyDataset, self).__init__()
        self.device = device
        self.fnames = glob.glob(os.path.join(img_path+"*.jpg"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = self.transforms(img)
        img = img.to(self.device)
        return img

    def __len__(self):
        return len(self.fnames)


def gradient_penality(discriminator, real, fake, device='cpu'):
    b, c, h, w = real.shape
    alpha = torch.randn((b, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = (real*alpha + fake * (1 - alpha)).requires_grad_(True)
    scores = discriminator(interpolated_images)

    gradient = torch.autograd.grad(inputs=interpolated_images,
                                   outputs=scores,
                                   grad_outputs=torch.ones_like(scores),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penality = torch.mean((gradient_norm - 1).square())
    return gradient_penality


if __name__ == "__main__":
    img_path = '/home/shixiaojia/dl/datasets/faces'
    MyDataset(img_path)
