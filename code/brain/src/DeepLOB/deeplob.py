"""
DeepLOB — faithful to Zhang, Zohren, Roberts (arXiv:1808.03668v6)
Matches Figure 3, Figure 4, Section IV-B, and Table III (~60k params).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_4x1(x):
    """Keras-style 'same' padding for a (4,1) kernel: pad 1 before, 2 after in time."""
    return F.pad(x, (0, 0, 1, 2))   # (left, right, top, bottom)


class InceptionModule(nn.Module):
    """Paper Figure 4: three parallel paths with 1x1 bottlenecks, then concat.
    Output channels = 3 * out_channels (not 4)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Path 1: 1x1 bottleneck → 3x1 temporal
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        # Path 2: 1x1 bottleneck → 5x1 temporal
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        # Path 3: MaxPool → 1x1
        self.path3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return torch.cat([self.path1(x), self.path2(x), self.path3(x)], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, num_horizons=3, num_classes=3):
        super().__init__()
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        F_ = 16  # paper uses 16 filters throughout CNN

        # --- Group 1: spatial squeeze 40→20, then 2 temporal denoise ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F_, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01))
        self.conv2 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),   # same-padded in forward()
            nn.LeakyReLU(0.01))
        self.conv3 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),
            nn.LeakyReLU(0.01))

        # --- Group 2: spatial squeeze 20→10, then 2 temporal denoise ---
        self.conv4 = nn.Sequential(
            nn.Conv2d(F_, F_, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01))
        self.conv5 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),
            nn.LeakyReLU(0.01))
        self.conv6 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),
            nn.LeakyReLU(0.01))

        # --- Group 3: spatial squeeze 10→1, then 2 temporal denoise ---
        self.conv7 = nn.Sequential(
            nn.Conv2d(F_, F_, (1, 10)),
            nn.LeakyReLU(0.01))
        self.conv8 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),
            nn.LeakyReLU(0.01))
        self.conv9 = nn.Sequential(
            nn.Conv2d(F_, F_, (4, 1)),
            nn.LeakyReLU(0.01))

        # --- Inception (3 paths × 32 filters = 96 channels) ---
        self.inception1 = InceptionModule(F_, 32)
        self.inception2 = InceptionModule(96, 32)

        # --- LSTM ---
        self.lstm = nn.LSTM(input_size=96, hidden_size=64,
                            num_layers=1, batch_first=True)

        # --- Output head ---
        self.fc = nn.Linear(64, num_horizons * num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)               # (B, 1, 100, 40)

        # Group 1
        x = self.conv1(x)                # (B, 16, 100, 20)
        x = self.conv2(_same_pad_4x1(x)) # (B, 16, 100, 20)
        x = self.conv3(_same_pad_4x1(x)) # (B, 16, 100, 20)

        # Group 2
        x = self.conv4(x)                # (B, 16, 100, 10)
        x = self.conv5(_same_pad_4x1(x)) # (B, 16, 100, 10)
        x = self.conv6(_same_pad_4x1(x)) # (B, 16, 100, 10)

        # Group 3
        x = self.conv7(x)                # (B, 16, 100, 1)
        x = self.conv8(_same_pad_4x1(x)) # (B, 16, 100, 1)
        x = self.conv9(_same_pad_4x1(x)) # (B, 16, 100, 1)

        # Inception
        x = self.inception1(x)           # (B, 96, 100, 1)
        x = self.inception2(x)           # (B, 96, 100, 1)

        # LSTM
        x = x.squeeze(3)                 # (B, 96, 100)
        x = x.permute(0, 2, 1)           # (B, 100, 96)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]    # (B, 64)

        # Classification
        out = self.fc(last_step)
        return out.view(-1, self.num_horizons, self.num_classes)