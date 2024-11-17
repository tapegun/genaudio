import torch
import torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, num_layers=10, num_classes=256):
        super(WaveNet, self).__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2**i)
            for i in range(num_layers)
        ])
        self.output_conv = nn.Conv1d(in_channels=1, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.dilated_convs:
            x = torch.relu(conv(x))
        x = self.output_conv(x)
        return x.squeeze(1)

