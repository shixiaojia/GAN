import os
import torch
import torch.nn as nn
from utils import MyDataset
from torch.utils.data import DataLoader
from network import Generator, Discriminator
from tqdm import tqdm
import argparse
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argsParser():
    parser = argparse.ArgumentParser(description='prepare for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training.')
    parser.add_argument('--n_epoch', type=int, default=100, help='num of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training.')
    parser.add_argument('--noisy_dim', type=int, default=128, help='dims for noisy.')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers for dataloader.')
    return parser.parse_args()


def train(args):
    gen = Generator(args.noisy_dim)
    dis = Discriminator()

    gen = gen.to(DEVICE)
    dis = dis.to(DEVICE)

    criterion = nn.BCELoss()

    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.learning_rate)
    dis_opt = torch.optim.Adam(dis.parameters(), lr=args.learning_rate)

    my_dataset = MyDataset(img_path='../faces/', device=DEVICE)

    dataloader = DataLoader(dataset=my_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
    for e in range(args.n_epoch):
        gen.train()
        dis.train()
        total_gen_loss = 0.
        total_dis_loss = 0.
        step = 0
        for idx, data in enumerate(tqdm(dataloader,  desc='Epoch {}: '.format(e))):
            data = data.to(DEVICE)
            N, *_ = data.shape
            noisy = torch.randn(N, args.noisy_dim).to(DEVICE)
            r_imgs = data
            r_label = torch.ones((N, )).to(DEVICE)
            f_label = torch.zeros((N, )).to(DEVICE)

            f_imgs = gen(noisy)

            r_logit = dis(r_imgs)
            f_logit = dis(f_imgs.detach())

            # discriminator loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_dis = (r_loss + f_loss)/2
            total_dis_loss += loss_dis
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()

            # train generator
            f_logit = dis(f_imgs)

            loss_gen = criterion(f_logit, r_label)

            total_gen_loss += loss_gen

            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

            step += 1

        gen = gen.eval()
        noisy = torch.randn(64, args.noisy_dim).to(DEVICE)
        images = gen(noisy)

        fname = './my_generated-images-{0:0=4d}.png'.format(e)
        save_image(images, fname, nrow=8)

        if not os.path.exists('./logs/{}'.format(e)):
            os.makedirs('./logs/{}'.format(e), exist_ok=True)

        if e % 10 == 0:
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt', exist_ok=True)
            torch.save(gen.state_dict(), './ckpt/gen_{}.pth'.format(e))
            torch.save(dis.state_dict(), './ckpt/dis_{}.pth'.format(e))


if __name__ == '__main__':
    args = argsParser()
    train(args)
