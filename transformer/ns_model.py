import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :]  # (seq_len, d_model)
        pe = pe.unsqueeze(1)         # (seq_len, 1, d_model)
        x = x + pe
        return x

class TransformerPooling(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.linear_in = nn.Linear(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # 平均池化压缩序列长度
        self.linear_out = nn.Linear(d_model, 3)

    def forward(self, x):
        # 输入形状: [582, 128, 3]
        x = self.linear_in(x)          # [582, 128, d_model]
        x = x.permute(1, 0, 2)         # [128, 582, d_model] (seq_len, batch, d_model)
        x = self.pos_encoder(x)        # 添加位置编码
        x = self.transformer_encoder(x) # [128, 582, d_model]
        x = x.permute(1, 0, 2)         # [582, 128, d_model]
        x = self.pool(x.permute(0, 2, 1))  # [582, d_model, 1]
        x = x.squeeze(-1)              # [582, d_model]
        x = self.linear_out(x)         # [582, 3]
        return x

class TransformerCLS(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 可学习的[CLS]标记
        self.linear_in = nn.Linear(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=1
        )
        self.linear_out = nn.Linear(d_model, 3)

    def forward(self, x):
        # 输入形状: [582, 128, 3]
        batch_size = x.size(0)
        x = self.linear_in(x)          # [582, 128, d_model]
        x = x.permute(1, 0, 2)         # [128, 582, d_model]
        # 添加[CLS]标记
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # [1, 582, d_model]
        x = torch.cat([cls_tokens, x], dim=0)  # [129, 582, d_model]
        x = self.pos_encoder(x)        # 添加位置编码
        x = self.transformer_encoder(x)

