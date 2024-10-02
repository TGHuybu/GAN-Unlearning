import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: 3 x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            # output: 3 x 64 x 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            # output: 64 x 32 x 32

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            # output: 128 x 16 x 16

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            # output: 256 x 8 x 8

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            # output: 512 x 4 x 4

            nn.Conv2d(in_channels=512, out_channels=768, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.01),
            # output: 768 x 2 x 2

            nn.Flatten(),
            #output : 768*2*2
            nn.Linear(in_features=768*2*2,out_features=1),
                 
            nn.Sigmoid()
            # output: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x)