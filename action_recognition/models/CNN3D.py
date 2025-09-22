import torch
import torch.nn as nn


class CNN3DModel2Conv(nn.Module):
    def __init__(self, **kwargs):
        super(CNN3DModel2Conv, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]

        self.conv1 = torch.nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

        self.fc1 = torch.nn.Sequential(
            nn.Linear(128 * (frames // 4) * (height // 4) * (width // 4), 256),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN3DModel3Conv(nn.Module):
    def __init__(self, **kwargs):
        super(CNN3DModel3Conv, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]

        self.conv1 = torch.nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.LeakyReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.LeakyReLU()
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()

        self.fc1 = torch.nn.Sequential(
            nn.Linear(128 * (frames // 8) * (height // 8) * (width // 8), 256),
            nn.Dropout(0.25),
            nn.LeakyReLU()
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
