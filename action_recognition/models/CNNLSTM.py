import torch
import torch.nn as nn


class CNNLSTMmodel(nn.Module):
    def __init__(self, **kwargs):
        super(CNNLSTMmodel, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        self.frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]

        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        flattened_size = 128 * (height // 16) * (width // 16)
        self.lstm = torch.nn.LSTM(input_size=18496, hidden_size=512, num_layers=1, batch_first=True)

        self.fc1 = torch.nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        hidden = None
        out = None
        for i in range(self.frames):
            x_slice = x[:, :, i]
            out = self.conv1(x_slice)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.flatten(out)
            out, hidden = self.lstm(out, hidden)
        x = self.fc1(out)
        return x


class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, **kwargs):
        super(CNNLSTMAttentionModel, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        self.frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()
        flattened_size = 128 * (height // 8) * (width // 8)
        self.lstm = torch.nn.LSTM(input_size=flattened_size, hidden_size=512, num_layers=1, batch_first=True)
        self.attention = torch.nn.Sequential(
            nn.Linear(512, flattened_size),
            nn.Dropout(0.25),
            nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )

        self.fc1 = torch.nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU()
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print("x.shape:", x.shape)
        hidden = None
        cell = None
        out = None
        # print("x.shape:", x.shape) [16, 3, 15, 128, 128]
        for i in range(self.frames):
            x_slice = x[:, :, i]
            # print("x_slice.shape:",x_slice.shape)
            out = self.conv1(x_slice)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.flatten(out)
            out = out.unsqueeze(1)
            if hidden is None:
                out, (hidden, cell) = self.lstm(out)
            else:
                attention = self.attention(hidden.squeeze(0)).unsqueeze(1)
                out = out * attention
                out, (hidden, cell) = self.lstm(out, (hidden, cell))
        x = self.fc1(out.squeeze(1))
        # print("x11.shape:", x.shape)
        x = self.fc2(x)
        # print("x22.shape:", x.shape)
        return x



import torch
import torch.nn as nn

class CNN6LSTMAttentionModel(nn.Module):
    def __init__(self, **kwargs):
        super(CNN6LSTMAttentionModel, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        self.frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),   # -> (B, 32, H, W)
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # -> (B, 32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),                # -> (B, 64, H/2, W/2)
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # -> (B, 64, H/4, W/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),               # -> (B, 128, H/4, W/4)
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                                            # -> (B, 128, H/8, W/8)

            nn.Conv2d(128, 128, kernel_size=3, padding=1),              # -> (B, 128, H/8, W/8)
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),              # -> (B, 128, H/8, W/8)
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),              # -> (B, 128, H/8, W/8)
            nn.LeakyReLU(),
        )

        self.flatten = nn.Flatten()
        flattened_size = 128 * (height // 8) * (width // 8)

        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=512, num_layers=1, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(512, flattened_size),
            nn.Dropout(0.25),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        hidden = None
        cell = None
        out = None
        for i in range(self.frames):
            x_slice = x[:, :, i]       # shape: [B, C, H, W]
            out = self.conv(x_slice)   # pass through 6 conv layers
            out = self.flatten(out)    # shape: [B, flattened_size]
            out = out.unsqueeze(1)     # shape: [B, 1, flattened_size]
            if hidden is None:
                out, (hidden, cell) = self.lstm(out)
            else:
                attention = self.attention(hidden.squeeze(0)).unsqueeze(1)
                out = out * attention
                out, (hidden, cell) = self.lstm(out, (hidden, cell))
        x = self.fc1(out.squeeze(1))
        x = self.fc2(x)
        return x
    
    
class CNNLSTMAttentionVideoModel(nn.Module):
    def __init__(self, **kwargs):
        super(CNNLSTMAttentionVideoModel, self).__init__()
        input_channels = kwargs["input_channels"]
        num_classes = kwargs["num_classes"]
        self.frames = kwargs["seq_len"]
        height = kwargs["height"]
        width = kwargs["width"]
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()
        flattened_size = 128 * (height // 8) * (width // 8)
        self.lstm = torch.nn.LSTM(input_size=flattened_size, hidden_size=512, num_layers=1, batch_first=True)
        self.attention = torch.nn.Sequential(
            nn.Linear(512, flattened_size),
            nn.Dropout(0.25),
            nn.Tanh(),
            torch.nn.Softmax(dim=1)
        )

        self.fc1 = torch.nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU()
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        hidden = None
        cell = None
        out = None
        # print("x.shape:", x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        # print("x.shape_after:", x.shape)11
        for i in range(self.frames):
            x_slice = x[:, :, i]
            # print("x_slice.shape:",x_slice.shape)
            out = self.conv1(x_slice)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.flatten(out)
            out = out.unsqueeze(1)
            if hidden is None:
                out, (hidden, cell) = self.lstm(out)
            else:
                attention = self.attention(hidden.squeeze(0)).unsqueeze(1)
                out = out * attention
                out, (hidden, cell) = self.lstm(out, (hidden, cell))
        x = self.fc1(out.squeeze(1))
        # print("x11.shape:", x.shape)
        x = self.fc2(x)
        # print("x22.shape:", x.shape)
        return x
