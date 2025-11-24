import torch.nn as nn
import torch
import numpy as np
import pandas as pd

class OptimizedCNN_Model(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout_rate, activation):
        """
        修改后的构造函数，接受超参数。

        Args:
            input_channels (int): 输入特征的维度。
            hidden_dim (int): 第一个卷积层的输出通道数（基础隐藏维度）。
                               后续卷积层通道数会基于此计算（例如减半）。
            dropout_rate (float): MLP 层使用的 Dropout 比率。
            activation (str): 要使用的激活函数的名称（例如 'relu', 'leaky_relu'）。
        """
        super(OptimizedCNN_Model, self).__init__()

        # ---- 参数化卷积层 ----
        # 使用 hidden_dim 作为第一个卷积层的输出通道数
        # 后续层通道数可以基于 hidden_dim 动态计算（例如减半）
        conv1_out_channels = hidden_dim
        conv2_out_channels = hidden_dim // 2
        conv3_out_channels = hidden_dim // 4

        # 确保通道数至少为 1 (避免 hidden_dim 过小时出现问题)
        conv1_out_channels = max(1, conv1_out_channels)
        conv2_out_channels = max(1, conv2_out_channels)
        conv3_out_channels = max(1, conv3_out_channels)

        self.conv1 = nn.Conv1d(input_channels, conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out_channels)

        self.conv2 = nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)

        self.conv3 = nn.Conv1d(conv2_out_channels, conv3_out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv3_out_channels)

        # ---- 参数化激活函数 ----
        self.act = self._get_activation(activation) # 使用辅助函数选择激活函数

        # ---- 参数化残差连接 ----
        # 残差连接的投影层需要匹配最后一个卷积层的输出通道数
        self.residual_projection = nn.Conv1d(input_channels, conv3_out_channels, kernel_size=1)

        # ---- 池化层 ----
        # 使用自适应平均池化层将序列维度压缩为1，得到固定大小的输出
        # 输出形状将是 (batch, conv3_out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # ---- 参数化 MLP 输出层 ----
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_rate), # 使用传入的 dropout_rate
            # MLP的输入维度是最后一个卷积层的输出通道数
            nn.Linear(conv3_out_channels, 1),
            nn.Sigmoid() # 保留 Sigmoid，因为 hy.py 中使用了 BCELoss
        )

    def _get_activation(self, activation_name):
        """根据名称返回激活函数实例"""
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'gelu':
            return nn.GELU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        else:
            print(f"Warning: Unsupported activation '{activation_name}'. Defaulting to ReLU.")
            return nn.ReLU() # 默认或抛出错误

    def forward(self, x):
        # 输入 x 的形状预期是 (batch, seq_len, features)
        # 卷积层需要 (batch, channels/features, seq_len)
        x = x.permute(0, 2, 1)  # 调整为 (batch, input_channels, seq_len)

        # 保存初始输入用于残差连接 (投影前)
        residual = x

        # 卷积块 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 卷积块 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 卷积块 3
        x = self.conv3(x)
        x = self.bn3(x)
        # 注意：在与残差相加之前，通常也应用激活函数
        x = self.act(x)

        # 残差连接
        residual_projected = self.residual_projection(residual)
        x = x + residual_projected
        # 有时在残差连接后也加一个激活函数
        x = self.act(x)

        # 全局平均池化
        # 输入 x 形状: (batch, conv3_out_channels, seq_len)
        # 输出 pooled_x 形状: (batch, conv3_out_channels, 1)
        pooled_x = self.global_avg_pool(x)
        # 展平池化后的输出以输入 MLP
        # 形状变为 (batch, conv3_out_channels)
        flattened_x = pooled_x.squeeze(-1) # 移除最后一个维度

        # MLP 输出
        final_out = self.mlp(flattened_x) # 输入 (batch, conv3_out_channels)
        final_out = torch.squeeze(final_out, dim=-1) # 输出 (batch,)

        return final_out

class CNN_Model(nn.Module):
    def __init__(self, input_channels):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.bigrucell = nn.GRU(1024, 64, num_layers=2, bidirectional=True)
        self.attention = nn.TransformerEncoderLayer(d_model=1024, nhead=4,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.attention, num_layers=4)
        self.transformer_linear = nn.Linear(1024, 128)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1):
        x1 = x1.permute(0, 2, 1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.act(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.act(x1)
        x1 = x1.permute(0, 2, 1)
        final_out = x1
        final_out = self.mlp(final_out)
        final_out = torch.squeeze(final_out, dim=-1)  # 只挤压最后一个维度

        return final_out

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):

        if isinstance(features, pd.DataFrame):
            # print("特征是DataFrame，使用.values提取NumPy数组")
            self.features = torch.tensor(features.values.astype(np.float32))
        elif isinstance(features, np.ndarray):
            # print("特征是NumPy数组，直接转换为tensor")
            self.features = torch.tensor(features.astype(np.float32))
        else:
            # print("特征是其他类型，尝试直接转换")
            try:
                self.features = torch.tensor(np.array(features, dtype=np.float32))
            except Exception as e:
                # print(f"转换特征时出错: {e}")
                # 如果出错，尝试另一种方法
                self.features = torch.tensor(list(features), dtype=torch.float32)

        # 同样处理标签
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = torch.tensor(labels.values, dtype=torch.float32)
        else:
            self.labels = torch.tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.features[idx]
        label = self.labels[idx]
        return data, label
