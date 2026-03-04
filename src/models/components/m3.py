from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.seq_of_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq_of_layers(x)


class M3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_size: int,
        num_classes: int,
        linear_in_features: int,
    ):
        super().__init__()

        layers = []
        current_in = in_channels

        for ch in channels:
            layers.append(
                ConvBlock(
                    kernel_size=kernel_size,
                    in_channels=current_in,
                    out_channels=ch,
                )
            )
            current_in = ch

        layers.extend([
            nn.Flatten(),
            nn.Linear(linear_in_features, num_classes),
            nn.BatchNorm1d(num_classes),
        ])

        self.seq_of_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq_of_layers(x)