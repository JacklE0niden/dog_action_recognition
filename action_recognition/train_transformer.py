import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from dataloader import ImageDataset
from models.CNN3D import CNN3DModel3Conv, CNN3DModel2Conv
from models.CNNLSTM import CNNLSTMmodel, CNNLSTMAttentionModel
from models.ViT import VisionTransformer
from models.ViViT import ViViT
from utils.utils import load_checkpoint, save_checkpoint
from evaluate import run_eval
import os

import argparse

# from torchinfo import summary

torch.manual_seed(42)



parser = argparse.ArgumentParser()
parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--scaledim", type=int, default=256, help="Dimension of embedding")
parser.add_argument("--depth", type=int, default=6, help="Number of transformer layers")
parser.add_argument("--device", type=str, default='cuda', help="Device")
args = parser.parse_args()

suffix = f"h{args.heads}_d{args.depth}_s{args.scaledim}"

train_images_name = "data/refined_Doge_dataset/data128.pkl"

dataset = ImageDataset(train_images_name)
data_size = len(dataset)
print("data_size:", data_size)

train_test_split = 0.75
train_size = round(data_size*0.75)
test_size = data_size - train_size
print(f"train size: {train_size}; test size: {test_size}")

generator = torch.Generator().manual_seed(63)
train_data, test_data = random_split(dataset, [train_size, test_size], generator=generator)
train_loader = data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=16, shuffle=True)

print("Data loaded!")
device = torch.device( args.device if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model = CNN3DModel3Conv(**dataset.input_shape).to(device)
# model = CNNLSTMAttentionModel(**dataset.input_shape).to(device)
# model = VisionTransformer(**dataset.input_shape).to(device)
model = ViViT(
    heads=args.heads,
    scaledim=args.scaledim,
    depth=args.depth,
    **dataset.input_shape
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# model = VisionTransformer(
#     input_channels=3,        # 输入视频的通道数（RGB视频是3）
#     num_classes=9,           # 类别数
#     seq_len=15,              # 视频帧数（例如：30帧）
#     height=128,              # 图像高度
#     width=128,               # 图像宽度
#     patch_size=16,           # 每个patch的大小（例如：16x16）
#     embedding_dim=768,       # 嵌入维度
#     num_heads=12,            # 注意力头数
#     feedforward_dim=2048,    # 前馈层的维度
#     num_layers=12            # Transformer编码器的层数
# ).to(device)
# model = CNNLSTMmodel(**dataset.input_shape).to(device)
print(model)

balanced_weight = torch.Tensor(dataset.balanced_weight).to(device)
loss_func = torch.nn.CrossEntropyLoss(weight=balanced_weight)
opt = torch.optim.Adam(model.parameters(), lr=3e-5)
writer = SummaryWriter(f"runs/loss_transformer_refined_{suffix}")    #tensorboard画图记录
step = 0

load_model = False   # 载入模型
save_model = True   # 储存模型
loss_count = []
acc_count = []
epochs = 100

# 创建并打开文件以保存训练日志
log_file = open(f"transformer_refined_{suffix}.txt", "w")
log_file.write("Epoch, Train Loss, Train Accuracy\n")  # 写入标题行

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, opt)    # 载入上个存档点

for epoch in range(1, epochs+1):
    step_loss = []
    step_accuracy = []
    for i, batch in enumerate(train_loader):
        x, y = batch["videos"].to(device, dtype=torch.float), batch["labels"].to(device)    # [batch_size, channel, seq_len, h, w] 的图片和 [1, 标签长度] 的标签
        batch_x = Variable(x)  # torch.Size([batch_size, channel, seq_len, h, w])
        batch_y = Variable(y)  # torch.Size([batch_size])
        
        # 获取最后输出
        output = model(batch_x)
        
        # 获取损失
        loss = loss_func(output, batch_y)
        
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上

        result = torch.max(output, 1)[1].cpu().numpy()
        accuracy = (result == batch_y.cpu().numpy()).mean()
        
        writer.add_scalar("Training loss", loss, global_step=step)  # tensorboard记录
        writer.add_scalar("Training acc", accuracy, global_step=step)  # tensorboard记录
        step += 1

        step_accuracy.append(accuracy)
        step_loss.append(loss.item())

    epoch_loss = sum(step_loss) / len(step_loss)
    epoch_acc = sum(step_accuracy) / len(step_accuracy)
    loss_count.append(epoch_loss)
    acc_count.append(epoch_acc)
    
    # 将每个epoch的输出写入日志文件
    log_file.write(f"{epoch:03d}, {epoch_loss:.3f}, {epoch_acc:.3f}\n")
    
    print('epoch {:03d}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, epoch_loss, epoch_acc))
    
    if epoch % 5 == 0:
        run_eval(test_loader, model, device, loss_func=loss_func)
    
    if epoch % 10 == 0 and epoch >= 10 and save_model:    # 存取模型和checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_{suffix}.pth.tar")
        torch.save(model, f"save/transformer_refined_{suffix}.pt")

if epochs:
    print('AvgLoss:\t', sum(loss_count)/len(loss_count))

run_eval(test_loader, model, device, loss_func=loss_func)
plt.figure('transformer_refined_Loss')
plt.plot(loss_count, label='Loss')
plt.legend()

output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, f"loss_transformer_refined_{suffix}.png"))

# 关闭日志文件
log_file.close()