import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 每个分支通过两层卷积压缩64维到1维
        self.object_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        self.relation_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        self.time_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        # 融合三个特征并生成最终输出
        self.combine_conv = nn.Conv1d(3, 1, kernel_size=1)

    def forward(self, object_emb, relation_emb, time_emb):
        # 输入形状均为 [582, 64, 200]
        object_feat = self.object_conv(object_emb)       # [582, 1, 200]
        relation_feat = self.relation_conv(relation_emb) # [582, 1, 200]
        time_feat = self.time_conv(time_emb)             # [582, 1, 200]
        # 拼接并融合
        combined = torch.cat([object_feat, relation_feat, time_feat], dim=1)  # [582, 3, 200]
        combined = self.combine_conv(combined)           # [582, 1, 200]
        return combined

class FeatureFusionTransformer(nn.Module):
    def __init__(self, d_model=200, nhead=8, num_layers=2):
        super().__init__()
        # 可学习的查询向量用于聚合全局信息
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, obj_emb, rel_emb, time_emb):
        # 拼接三个嵌入张量：[batch_size, 64*3, 200]
        combined = torch.cat([obj_emb, rel_emb, time_emb], dim=1)
        batch_size = combined.size(0)
        
        # 扩展可学习查询向量到批次大小
        query = self.query.expand(batch_size, -1, -1)  # [batch_size, 1, 200]
        
        # 将查询向量与输入序列拼接
        transformer_input = torch.cat([query, combined], dim=1)  # [batch_size, 1+192, 200]
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(transformer_input)
        
        # 提取查询位置的输出作为全局特征
        global_feature = encoded[:, 0:1, :]  # [batch_size, 1, 200]
        return global_feature





class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)  # 添加位置编码
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, d_model=200, nhead=5, num_layers=3, dim_feedforward=800):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 可学习的特殊 Token [1, 1, d_model]
        self.special_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码模块
        self.pos_encoder = PositionalEncoding(d_model, max_len=65)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # PyTorch 1.9+ 支持 batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 [batch_size, seq_len=64, d_model=200]
        Returns:
            输出张量，形状 [batch_size, 1, d_model=200]
        """
        batch_size = x.size(0)
        
        # 插入特殊 Token [batch_size, 1, d_model]
        special_token = self.special_token.expand(batch_size, -1, -1)
        x = torch.cat([special_token, x], dim=1)  # [batch_size, 65, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过 Transformer 编码器
        x = self.transformer(x)  # [batch_size, 65, d_model]
        
        # 提取特殊 Token 的输出
        output = x[:, 0:1, :]  # [batch_size, 1, d_model]
        return output


class CrossAttentionFusion(nn.Module):
    def __init__(self, noise_dim, history_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # 将噪声映射为查询向量
        self.q_proj = nn.Linear(noise_dim, noise_dim)
        # 将历史条件映射为键和值
        self.k_proj = nn.Linear(history_dim, noise_dim)
        self.v_proj = nn.Linear(history_dim, noise_dim)
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=noise_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 确保输入为 [B, seq_len, dim]
        )
        # 自适应层归一化
        self.norm = nn.LayerNorm(noise_dim)
        # 残差连接的可学习权重
        self.res_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, base_noise, c_history):
        """
        base_noise: [B, noise_seq_len, d_noise]
        c_history: [B, history_seq_len, d_history]
        """
        # 投影查询（Query）
        Q = self.q_proj(base_noise)  # [B, noise_seq_len, d_noise]
        
        # 投影键（Key）和值（Value）
        K = self.k_proj(c_history)   # [B, history_seq_len, d_noise]
        V = self.v_proj(c_history)   # [B, history_seq_len, d_noise]
        
        # 计算交叉注意力（允许不同序列长度）
        attn_output, _ = self.attention(
            query=Q,
            key=K,
            value=V,
            need_weights=False
        )  # [B, noise_seq_len, d_noise]
        
        # 残差连接 + 层归一化
        cond_noise = self.norm(base_noise + self.res_weight * attn_output)
        return cond_noise

class GateUnitExpand(nn.Module):
    """
    拓展的gate单元，支持自由的因素大小。
    """

    def __init__(self, factor_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_W = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, object_embeddings, time_embeddings_re, relation_embeddings, mask):
        """
        通过类似GRU的门控机制更新实体表示
        """
        factors = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)
        # 计算门, 计算门时考虑到查询的关系
        update_value, reset_value = self.gate_W(factors).chunk(2, dim=-1)
        # 计算候选隐藏表示
#        print(message.shape)
#        print(reset_value.shape)
#        print(hidden_state.shape)
        mask_expanded = mask.unsqueeze(-1)
        scale = 1 + self.sigmoid(mask_expanded)
#        hidden_state = hidden_state * (1 + mask_expanded)
        hidden_candidate = self.hidden_trans(torch.cat([time_embeddings_re, reset_value * relation_embeddings * scale], dim=-1))
        hidden_state = (1 - update_value) * object_embeddings + update_value * hidden_candidate
        return hidden_state
