import torch.nn as nn
import torch
import numpy as np
import pandas as pd

from mamba_ssm import Mamba  # 需要安装 mamba-ssm 库

class OptimizedBiGRU_Model(nn.Module):
    def __init__(self, input_channels):
        super(OptimizedBiGRU_Model, self).__init__()
        # 双向 GRU 层
        self.bigrucell = nn.GRU(input_channels, 32, num_layers=2, bidirectional=True)
        # 残差连接的投影层
        self.residual_projection = nn.Linear(input_channels, 64)  # 双向 GRU 输出为 32*2=64
        # MLP 输出层
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 保存输入用于残差连接
        residual = x
        # GRU 处理
        x, _ = self.bigrucell(x)
        # 残差连接
        residual = self.residual_projection(residual)
        x = x + residual
        # MLP 输出
        final_out = self.mlp(x)
        return final_out


class OptimizedCNN_Model(nn.Module):
    def __init__(self, input_channels):
        super(OptimizedCNN_Model, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(input_channels,256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(64,32, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        # 批量归一化层
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # self.bn4 = nn.BatchNorm1d(128)
        # self.bn5 = nn.BatchNorm1d(32)

        # 激活函数
        self.act = nn.ReLU()
        # 残差连接的投影层
        self.residual_projection = nn.Conv1d(input_channels, 64, kernel_size=1)
        # MLP 输出层
        self.mlp = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 保存输入用于残差连接
        residual = x.permute(0, 2, 1)  # 调整为 (batch, channels, seq_len)
        # print("Input after permute:", torch.isnan(residual).any(), torch.isinf(residual).any())
        # 卷积层处理
        x = x.permute(0, 2, 1)  # 调整为 (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.act(x)
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.act(x)
        # 残差连接
        residual = self.residual_projection(residual)
        x = x + residual
        # 调整回 (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        # x = x.mean(dim=1)  # 平均池化，得到 (batch, 64)
        # MLP 输出
        # final_out = self.mlp(x)
        # final_out = torch.squeeze(final_out, dim=-1)
        final_out = self.mlp(x.view(x.size(0), -1))  # 展平后输入 MLP
        final_out = torch.squeeze(final_out, dim=-1)
        return final_out


class Mamba_Model(nn.Module):
    def __init__(self, input_channels, d_model=128, d_state=16, d_conv=4, expand=2):
        """
        Mamba 模型类，用于序列分类任务，与现有模型一致。

        参数:
            input_channels (int): 输入通道数，与特征维度对应
            d_model (int): Mamba 模型的内部维度，默认 128
            d_state (int): 状态空间维度，默认 16
            d_conv (int): 卷积核大小，默认 4
            expand (int): 特征扩展因子，默认 2
        """
        super(Mamba_Model, self).__init__()
        # 输入投影层：将 input_channels 映射到 d_model
        self.input_projection = nn.Linear(input_channels, d_model)
        # Mamba 层，使用官方 mamba-ssm 实现
        self.mamba = Mamba(
            d_model=d_model,  # 模型维度
            d_state=d_state,  # SSM 状态维度
            d_conv=d_conv,  # 卷积核大小
            expand=expand  # 扩展因子
        )
        # 输出 MLP 层，与现有模型一致
        self.output_mlp = nn.Sequential(
            nn.Dropout(0.2),  # Dropout 防止过拟合，与现有模型一致
            nn.Linear(d_model, 1),  # 映射到单一输出
            nn.Sigmoid()  # 输出概率值
        )

    def forward(self, x):
        """
        前向传播。
        输入:
            x (torch.Tensor): 形状为 (batch_size, sequence_length, input_channels) 的输入张量
        输出:
            torch.Tensor: 形状为 (batch_size, sequence_length) 的输出张量
        """
        # 输入形状：(batch_size, sequence_length, input_channels)
        if x.dim() == 2:  # 如果输入是 (batch_size, input_channels)
            x = x.unsqueeze(1)  # 变为 (batch_size, 1, input_channels)

        x = self.input_projection(x)  # 投影到 (batch_size, sequence_length, d_model)
        mamba_out = self.mamba(x)  # Mamba 处理：(batch_size, sequence_length, d_model)
        output = self.output_mlp(mamba_out)  # MLP 处理：(batch_size, sequence_length, 1)
        output = torch.squeeze(output, dim=-1)  # 移除多余维度：(batch_size, sequence_length)
        return output

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
        final_out = torch.squeeze(final_out)
        return final_out

class BiGRU_Model(nn.Module):
    def __init__(self, input_channels):
        super(BiGRU_Model, self).__init__()
        self.bigrucell = nn.GRU(input_channels, 32, num_layers=2, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x2):
        x2, _ = self.bigrucell(x2)
        final_out = x2
        final_out = self.mlp(final_out)
        final_out = torch.squeeze(final_out)
        # return final_out, x2
        return final_out

class AttTransform_Model(nn.Module):
    def __init__(self, input_channels):
        super(AttTransform_Model, self).__init__()
        self.input_linear = nn.Linear(input_channels, 1280)  # 新增：将 2302 映射到 1280
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.bigrucell = nn.GRU(1024, 64, num_layers=2, bidirectional=True)
        self.attention = nn.TransformerEncoderLayer(d_model=1280, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.attention, num_layers=4)
        self.transformer_linear = nn.Linear(1280, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_linear(x)  # 新增：先将维度从 2302 映射到 1280
        x = self.transformer_encoder(x)
        x = self.transformer_linear(x)

        final_out = x
        final_out = self.mlp(final_out)
        final_out = torch.squeeze(final_out)
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

class AdvancedCNN_Model(nn.Module):
    def __init__(self, input_channels):
        super(AdvancedCNN_Model, self).__init__()
        # 第一层卷积：提取基础特征
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        # 第二层卷积：提取高级特征
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        # 残差连接：调整通道数并稳定特征
        self.residual = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128)
        )
        # MLP 层：分类
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),  # 降低 Dropout 率，适度正则化
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        residual = self.residual(x)  # 残差分支
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))  # 不加激活，留给残差加和后处理
        x = x + residual  # 添加残差
        x = self.act(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 128)
        out = self.mlp(x)  # (batch, seq_len, 1)
        return torch.squeeze(out, dim=-1)  # (batch, seq_len)
