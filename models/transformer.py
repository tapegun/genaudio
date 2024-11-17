import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTransformer(nn.Module):
    def __init__(self, num_classes=256, d_model=128, num_layers=6, nhead=8):
        super(AudioTransformer, self).__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1)]
        x = self.transformer(x)
        return self.fc_out(x)

