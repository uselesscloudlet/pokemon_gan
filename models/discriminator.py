import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, channel_n, fm_dis):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (channel_n) x 64 x 64
            nn.Conv2d(channel_n, fm_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fm_dis) x 32 x 32
            nn.Conv2d(fm_dis, fm_dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fm_dis*2) x 16 x 16
            nn.Conv2d(fm_dis * 2, fm_dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fm_dis*4) x 8 x 8
            nn.Conv2d(fm_dis * 4, fm_dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fm_dis*8) x 4 x 4
            nn.Conv2d(fm_dis * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)