import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.view(-1, 3, 4, 4)
        return x


class Generator(nn.Module):
    def __init__(self, in_size):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_features=in_size,out_features=4*4*3)
        self.reshape = Reshape()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.reshape(x)
        x = self.conv(x)
        return x


class Generator_v2(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator_v2, self).__init__()
        self.gen = torch.nn.Sequential(
            #imgsize: 4 x 4
            self._block(in_channels=channels_noise, out_channels=features_g * 16, kernel_size=(4, 4),
                        stride=(1, 1), padding=0),
            # imgsize: 8 x 8
            self._block(in_channels=features_g * 16, out_channels=features_g * 8, kernel_size=(4, 4), stride=(2, 2),
                        padding=1),
            # imgsize: 16 x 16
            self._block(in_channels=features_g * 8, out_channels=features_g * 4, kernel_size=(4, 4), stride=(2, 2),
                        padding=1),
            # imgsize: 32 x 32
            self._block(in_channels=features_g * 4, out_channels=features_g * 2, kernel_size=(4, 4), stride=(2, 2),
                        padding=1),
            # imgsize: N x 3 x 64 x 64
            torch.nn.ConvTranspose2d(
                in_channels=features_g * 2, out_channels=channels_img, kernel_size=(4, 4), stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU()
        )
        return self.conv

    def forward(self, input):
        x = self.gen(input)
        return x


class conv_bn_relu(nn.Module):
    def __init__(self, in_size, out_size):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 5, 2, 2),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    '''
    shape (N, 3, 64, 64)
    '''
    def __init__(self, in_size=3, size=64):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, size, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_bn_relu(size, 2*size),
            conv_bn_relu(2*size, 4*size),
            conv_bn_relu(4*size, 8*size),
            nn.Conv2d(8*size, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1)
        return x


class Discriminator_v2(torch.nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator_v2, self).__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channels_img,out_channels=features_d,kernel_size=(4,4),stride=(2,2),padding=(1,1)
            ),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True),
            self._block(in_channels=features_d,out_channels=features_d * 2,kernel_size=(4,4),stride=(2,2),
                        padding=(1,1)),
            self._block(in_channels=features_d * 2, out_channels=features_d * 4, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            self._block(in_channels=features_d * 4, out_channels=features_d * 8, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            torch.nn.Conv2d(in_channels=features_d*8,out_channels=1,kernel_size=(4,4),stride=(2,2),padding=0)
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            #affine=True:一个布尔值，当设置为True时，该模块具有可学习的仿射参数，以与批量规范化相同的方式初始化。默认值：False。
            torch.nn.InstanceNorm2d(num_features=out_channels,affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        return self.conv

    def forward(self, input):
        x = self.disc(input)
        return x

# if __name__ == '__main__':
    # input = torch.randn(10, 128)
    # gen = Generator(128)
    # out = gen(input)
    # print(out.shape)

    # dis = Discriminator()
    #
    # input = torch.randn(10, 3, 64, 64)
    #
    # out = dis(input)
    # print(out.shape)