import torch.nn as nn
from helper import batch_size


class Visual_encoder(nn.Module):
    def __init__(self):
        super(Visual_encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=batch_size, out_channels=100, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.Conv2d(
                in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(
            in_channels=100, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        # self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        print("Shape of the Input in VGG network:-", x.shape)

        x = self.conv1(x.permute(1, 0, 2, 3))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.upsample1(x)
        # x=self.upsample2(x)
        # x=self.upsample3(x)
        return x
