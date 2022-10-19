import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_gpu, lat_vec, fm_gen, channel_n):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(lat_vec, fm_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm_gen * 8),
            nn.ReLU(True),
            # state size. (fm_gen*8) x 4 x 4
            nn.ConvTranspose2d(fm_gen * 8, fm_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_gen * 4),
            nn.ReLU(True),
            # state size. (fm_gen*4) x 8 x 8
            nn.ConvTranspose2d(fm_gen * 4, fm_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_gen * 2),
            nn.ReLU(True),
            # state size. (fm_gen*2) x 16 x 16
            nn.ConvTranspose2d(fm_gen * 2, fm_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_gen),
            nn.ReLU(True),
            # state size. (fm_gen) x 32 x 32
            nn.ConvTranspose2d(fm_gen, channel_n, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (channel_n) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
