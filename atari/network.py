import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # x shape should be: (batch_size, 4, 84, 84)
        x = x / 255.0  # normalize pixel values from 0-255 to 0-1
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
