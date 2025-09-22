import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__()
        self.input_channels = kwargs["input_channels"]
        self.num_classes = kwargs["num_classes"]
        self.seq_len = kwargs["seq_len"]  # 视频序列长度（即帧数）
        self.height = kwargs["height"]
        self.width = kwargs["width"]
        # [16, 3, 15, 128, 128]
        # Patch embedding: 将每一帧图像切割成小块并嵌入
        patch_size = kwargs["patch_size"]  # Patch size (e.g., 16x16)
        self.patch_size = patch_size
        self.num_patches = (self.height // patch_size) * (self.width // patch_size)
        
        self.patch_embedding = nn.Conv2d(in_channels=self.input_channels,
                                         out_channels=kwargs["embedding_dim"],
                                         kernel_size=patch_size, 
                                         stride=patch_size)
        
        # Transformer Encoder Layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kwargs["embedding_dim"], 
                nhead=kwargs["num_heads"], 
                dim_feedforward=kwargs["feedforward_dim"]
            ),
            num_layers=kwargs["num_layers"]
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(kwargs["embedding_dim"], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 输入 x 的形状应为 (batch_size, channels, seq_len, height, width) [16, 3, 15, 128, 128]
        batch_size = x.shape[0]
        
        # 将视频序列的每一帧分割成小块并嵌入
        frame_embeddings = []
        for i in range(self.seq_len):
            x_slice = x[:, :, i, :, :]  # 获取视频中的第 i 帧（形状：batch_size, channels, height, width）
            frame_embedding = self.patch_embedding(x_slice)  # Output shape: (batch_size, embedding_dim, num_patches, 1)
            frame_embeddings.append(frame_embedding)

        # 将每个帧的嵌入拼接在一起（形状：batch_size, seq_len, num_patches, embedding_dim）
        x = torch.stack(frame_embeddings, dim=1)  # Output shape: (batch_size, seq_len, embedding_dim, num_patch_side, num_patch_side)
        print("stack_shape:", x.shape)
        
        # 将每一帧的 num_patches 维度展平（将每个 patch 作为一个 token）
        x = x.flatten(3)  # Output shape: (batch_size, seq_len, num_patches * embedding_dim)
        
        # 使用 Transformer Encoder 进行特征提取
        x = x.transpose(0, 1)  # Transpose to (seq_len, batch_size, num_patches * embedding_dim)
        x = x.flatten(2)
        print("x0.shape:", x.shape)
        x = self.encoder(x)  # (seq_len, batch_size, num_patches * embedding_dim)
        
        # 分类头部分
        x = x.mean(dim=0)  # 对所有 seq_len 的输出进行平均池化
        x = self.fc(x)
        
        return x

# class VisionTransformer(nn.Module):
#     def __init__(self, **kwargs):
#         super(VisionTransformer, self).__init__()
        
#         # 从 kwargs 中获取超参数
#         self.input_channels = kwargs["input_channels"]
#         self.num_classes = kwargs["num_classes"]
#         self.seq_len = kwargs["seq_len"]  # 视频序列长度（即帧数）
#         self.height = kwargs["height"]
#         self.width = kwargs["width"]
#         self.embed_dim = kwargs.get("embed_dim", 768)  # 默认嵌入维度为 768
#         self.num_heads = kwargs.get("num_heads", 12)  # 默认头数为 12
#         self.num_layers = kwargs.get("num_layers", 12)  # 默认 Transformer 层数为 12
#         self.dropout = kwargs.get("dropout", 0.1)  # 默认丢弃率为 0.1

#         # 1. Patch Embedding
#         patch_size = kwargs["patch_size"]  # Patch size (e.g., 16x16)
#         self.patch_size = patch_size
#         self.proj = nn.Conv3d(in_channels=self.input_channels,
#                               out_channels=self.embed_dim,
#                               kernel_size=(self.patch_size, self.patch_size, self.patch_size),
#                               stride=(self.patch_size, self.patch_size, self.patch_size))

#         # 2. Transformer Encoder
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
#                                                         nhead=self.num_heads,
#                                                         dim_feedforward=2048,
#                                                         dropout=self.dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

#         # 3. Classification head
#         self.fc = nn.Linear(self.embed_dim, self.num_classes)

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 1. Patch Embedding
#         x = self.proj(x)  # Shape: [batch_size, embed_dim, seq_len/patch_size, height/patch_size, width/patch_size]
#         x = x.flatten(2)  # Flatten the spatial dimensions [batch_size, embed_dim, patches]

#         # 2. Prepare input for transformer
#         x = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]

#         # 3. Transformer Encoder
#         x = self.transformer_encoder(x)  # Output shape: [batch_size, seq_len, embed_dim]

#         # 4. Classification head
#         x = x.mean(dim=1)  # Taking the mean across the sequence length dimension
#         x = self.fc(x)

#         return x