from torch import nn

class Net(nn.Module):

    def __init__(self, in_channels, num_class):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 600),
            nn.Dropout(0.25),
            nn.Linear(600, 120),
            nn.Linear(120, num_class)
        )
    def forward(self, x):
        return self.layers(x)
