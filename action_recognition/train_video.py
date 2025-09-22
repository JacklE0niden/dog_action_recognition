import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNNLSTM import CNNLSTMAttentionVideoModel, CNNLSTMAttentionModel # 模型类
from dataloader import VideoDataset  # 数据集类
import pickle
import os
from tqdm import tqdm

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 20
learning_rate = 1e-4
max_seq_len = 30
frame_rate = 30
video_res = (224, 224)
pkl_path = 'train_val_data.pkl'  # 训练数据
model_save_path = 'models/CNNLSTM.pth'

# 加载数据集
train_dataset = VideoDataset(pkl_path=pkl_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = CNNLSTMAttentionVideoModel(
    input_channels=train_dataset.video_info["input_channels"],
    num_classes=len(train_dataset.label_map),
    seq_len=max_seq_len,
    height=video_res[0],
    width=video_res[1]
).to(device)

# 类别平衡
class_weights = torch.tensor(train_dataset.balanced_weight, dtype=torch.float32).to(device)
print("class_weights.shape:", class_weights.shape)
# 定义损失函数（加权交叉熵损失）
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            print("videos.shape:", videos.shape)
            outputs = model(videos)
            
            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 每个epoch保存模型
        torch.save(model.state_dict(), model_save_path)

# 训练模型
train(model, train_loader, criterion, optimizer, epochs)