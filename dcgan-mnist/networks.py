import torch.nn as nn

class Generator(nn.Module):
    # version 1 (original)
    def __init__(self, version=2, nz=128):
        super(Generator, self).__init__()

        self._nz = nz

        if version == 1:
            self._version = 1
            self.model = nn.Sequential(
                nn.ConvTranspose2d(nz, 128, 4, 2, 0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 4 x 4

                nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 7 x 7

                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                # state size. 32 x 14 x 14

                nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. 1 x 28 x 28
            )
            
        # version 2 (more layers)
        elif version == 2:
            self._version = 2
            self.model = nn.Sequential(
                # 1st Layer (nz → 128)
                nn.ConvTranspose2d(nz, 128, 4, 2, 0, bias=False),  # Output: (128, 4, 4)
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                # 2nd Layer (128 → 64)
                nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=False),  # Output: (64, 7, 7)
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # 3rd Layer (64 → 32)
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output: (32, 14, 14)
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                # New Extra Layer (32 → 16)
                nn.ConvTranspose2d(32, 16, 3, 1, 1, bias=False),  # Output: (16, 14, 14)
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                # 4th Layer (16 → 1)
                nn.ConvTranspose2d(16, 1, 4, 2, 1, bias=False),  # Output: (1, 28, 28)
                nn.Tanh()
            )

        # version 3 (even more complex)
        # NOTE: not good yet
        elif version == 3:
            self._version = 3
            self.model = nn.Sequential(
                # 1st Layer (nz → 128)
                nn.ConvTranspose2d(nz, 145, 4, 2, 0, bias=False),  # Output: (128, 4, 4)
                nn.BatchNorm2d(145),
                nn.ReLU(True),

                # 2nd Layer (128 → 64)
                nn.ConvTranspose2d(145, 105, 3, 2, 1, bias=False),  # Output: (64, 7, 7)
                nn.BatchNorm2d(105),
                nn.ReLU(True),

                # 3rd Layer (64 → 32)
                nn.ConvTranspose2d(105, 60, 4, 2, 1, bias=False),  # Output: (32, 14, 14)
                nn.BatchNorm2d(60),
                nn.ReLU(True),

                # New Extra Layer (32 → 16)
                nn.ConvTranspose2d(60, 30, 3, 1, 1, bias=False),  # Output: (16, 14, 14)
                nn.BatchNorm2d(30),
                nn.ReLU(True),

                # 4th Layer (16 → 1)
                nn.ConvTranspose2d(30, 1, 4, 2, 1, bias=False),  # Output: (1, 28, 28)
                nn.Tanh()
            )


    def forward(self, input):
        return self.model(input)
    

class Discriminator(nn.Module):
    def __init__(self, version=0):
        super(Discriminator, self).__init__()

        if version not in [3]:
            self.model = nn.Sequential(
                # input is 1 x 28 x 28

                nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 32 x 14 x 14

                nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 64) x 7 x 7

                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 128 x 4 x 4

                nn.Conv2d(128, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        # version 3 (match with gen)
        else:
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                # 1st Layer (Input: 1 × 28 × 28 → 32 × 14 × 14)
                nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                # 2nd Layer (32 → 64)
                nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # Output: (64, 7, 7)
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                # 3rd Layer (64 → 128)
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # Output: (128, 4, 4)
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # New Extra Layer (128 → 256)
                nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Output: (256, 4, 4)
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # 4th Layer (256 → 1)
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # Output: (1, 1, 1)
                nn.Sigmoid()
            )


    def forward(self, input):
        return self.model(input)
    

class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv = nn.Sequential(
            # conv 1
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 2
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # convert to a feature vector
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    
    def extract_features(self, x):
        return self.conv(x)


    def forward(self, x):
        features = self.conv(x)
        predictions = self.head(features)
        return predictions


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)