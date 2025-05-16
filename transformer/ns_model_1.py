import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.combine_conv = nn.Conv1d(3, 1, kernel_size=1)

    def forward(self, object_emb, relation_emb, time_emb):
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
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, obj_emb, rel_emb, time_emb):
        combined = torch.cat([obj_emb, rel_emb, time_emb], dim=1)
        batch_size = combined.size(0)
        
        query = self.query.expand(batch_size, -1, -1)  # [batch_size, 1, 200]
        
        transformer_input = torch.cat([query, combined], dim=1)  # [batch_size, 1+192, 200]
        
        encoded = self.transformer_encoder(transformer_input)
        
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
        
        self.special_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=65)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # PyTorch 1.9+ 支持 batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        special_token = self.special_token.expand(batch_size, -1, -1)
        x = torch.cat([special_token, x], dim=1)  # [batch_size, 65, d_model]
        
        x = self.pos_encoder(x)
        
        x = self.transformer(x)  # [batch_size, 65, d_model]
        output = x[:, 0:1, :]  # [batch_size, 1, d_model]
        return output


class CrossAttentionFusion(nn.Module):
    def __init__(self, noise_dim, history_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.q_proj = nn.Linear(noise_dim, noise_dim)

        self.k_proj = nn.Linear(history_dim, noise_dim)
        self.v_proj = nn.Linear(history_dim, noise_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=noise_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  
        )

        self.norm = nn.LayerNorm(noise_dim)

        self.res_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, base_noise, c_history):
        """
        base_noise: [B, noise_seq_len, d_noise]
        c_history: [B, history_seq_len, d_history]
        """
        Q = self.q_proj(base_noise)  # [B, noise_seq_len, d_noise]
        
        K = self.k_proj(c_history)   # [B, history_seq_len, d_noise]
        V = self.v_proj(c_history)   # [B, history_seq_len, d_noise]
        
        attn_output, _ = self.attention(
            query=Q,
            key=K,
            value=V,
            need_weights=False
        )  # [B, noise_seq_len, d_noise]
        
        cond_noise = self.norm(base_noise + self.res_weight * attn_output)
        return cond_noise

class GateUnitExpand(nn.Module):

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

        factors = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)

        update_value, reset_value = self.gate_W(factors).chunk(2, dim=-1)

#        print(message.shape)
#        print(reset_value.shape)
#        print(hidden_state.shape)
        mask_expanded = mask.unsqueeze(-1)
        scale = 1 + self.sigmoid(mask_expanded)
#        hidden_state = hidden_state * (1 + mask_expanded)
        hidden_candidate = self.hidden_trans(torch.cat([time_embeddings_re, reset_value * relation_embeddings * scale], dim=-1))
        hidden_state = (1 - update_value) * object_embeddings + update_value * hidden_candidate
        return hidden_state
