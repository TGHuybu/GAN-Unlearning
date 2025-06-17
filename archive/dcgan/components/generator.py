import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size=512):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=2048, bias=False),
            nn.LeakyReLU(0.01),
            nn.Unflatten(1, (2048, 1, 1)),  

            # output : 1024 x 1 x 1
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.01),
            # output : 768 x 2 x 2
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            #output : 512 x 4 x 4

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            # output: 128 x 8 x 8
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            # output : 256 x 16 x 16

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            # output : 128 x 32 x 32

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            # output : 128 x 64 x 64

            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
            # output: 3 x 128 x 128            
        )

    def forward(self, x):
        return self.main(x)