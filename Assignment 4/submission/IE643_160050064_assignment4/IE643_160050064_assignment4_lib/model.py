## AUTHOR: Vamsi Krishna Reddy Satti


import torch
from torch import nn


def _init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, out_channels, output_size, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, output_size * 8, 4, 1, 0),
            nn.BatchNorm2d(output_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(output_size * 8, output_size * 4, 4, 2, 1),
            nn.BatchNorm2d(output_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(output_size * 4, output_size * 2, 4, 2, 1),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(output_size * 2, output_size, 4, 2, 1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(output_size, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(_init)

    def forward(self, inp):
        inp = inp[..., None, None]
        out = self.main(inp)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, input_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, input_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_size, input_size * 2, 4, 2, 1),
            nn.BatchNorm2d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_size * 2, input_size * 4, 4, 2, 1),
            nn.BatchNorm2d(input_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_size * 4, input_size * 8, 4, 2, 1),
            nn.BatchNorm2d(input_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_size * 8, 1, 4, 1, 0, bias=False),
        )
        self.apply(_init)

    def forward(self, inp):
        out = self.main(inp)
        out = out.view(-1)
        return out
