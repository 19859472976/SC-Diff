import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_diffusion_step, filter_type="dual_random_walk"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_diffusion_step = max_diffusion_step

        # 门控参数（重置门+更新门）
        self.gate = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)

        # 候选状态参数
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # 扩散卷积核
        self.diffusion_conv = DiffusionConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            max_step=max_diffusion_step,
            filter_type=filter_type
        )

    def forward(self, x_t, h_prev, adj):
        """
        :param x_t: 当前时刻输入 (batch_size, num_nodes, input_dim)
        :param h_prev: 前一时刻隐藏状态 (batch_size, num_nodes, hidden_dim)
        :param adj: 邻接矩阵或动态图结构 (num_nodes, num_nodes)
        :return h_next: 下一时刻隐藏状态
        """
        combined = torch.cat([x_t, h_prev], dim=-1)  # (B, N, input+hidden)

        # 门控计算
        gate_vals = torch.sigmoid(self.gate(combined))  # (B, N, 2*hidden)
        r_gate, u_gate = torch.chunk(gate_vals, 2, dim=-1)

        # 空间扩散卷积
        conv_h = self.diffusion_conv(h_prev, adj)  # (B, N, hidden)

        # 候选状态
        candidate_input = torch.cat([x_t, r_gate * conv_h], dim=-1)
        c_t = torch.tanh(self.candidate(candidate_input))

        # 状态更新
        h_next = u_gate * h_prev + (1 - u_gate) * c_t

        return h_next


class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_step, filter_type):
        super().__init__()
        self.max_step = max_step
        self.filter_type = filter_type

        # 可学习扩散系数
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_channels, out_channels))
            for _ in range(max_step + 1)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x, adj):
        """
        :param x: 输入特征 (B, N, C_in)
        :param adj: 邻接矩阵 (N, N)
        :return: 扩散卷积结果 (B, N, C_out)
        """
        batch_size, num_nodes, _ = x.shape
        x = x.reshape(-1, num_nodes, x.size(-1))  # (B, N, C_in)

        # 预计算扩散矩阵
        adj_powers = self._get_diffusion_matrices(adj)  # List[(N, N)]

        # 多阶扩散聚合
        output = torch.zeros_like(x)
        for k in range(self.max_step + 1):
            x_trans = torch.matmul(adj_powers[k], x)  # (B, N, C_in)
            output += torch.matmul(x_trans, self.weights[k])  # (B, N, C_out)

        return output

    def _get_diffusion_matrices(self, adj):
        # 生成扩散矩阵序列
        adj_powers = [torch.eye(adj.size(0), device=adj.device)]
        for _ in range(self.max_step):
            adj_powers.append(torch.mm(adj_powers[-1], adj))
        return adj_powers


class SpatioTemporalBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, diffusion_steps):
        super().__init__()
        self.dcgru_cell = DCGRUCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_diffusion_step=diffusion_steps
        )

    def forward(self, x_seq, adj):
        """
        :param x_seq: 时序输入 (T, B, N, D_in)
        :param adj: 动态邻接矩阵 (T, N, N) 或静态 (N, N)
        :return outputs: 输出序列 (T, B, N, D_out)
        """
        batch_size, num_nodes = x_seq.size(1), x_seq.size(2)
        h = torch.zeros(batch_size, num_nodes, self.dcgru_cell.hidden_dim).to(x_seq.device)

        outputs = []
        for t in range(x_seq.size(0)):
            h = self.dcgru_cell(x_seq[t], h, adj)  # adj可动态变化
            outputs.append(h)

        return torch.stack(outputs)  # (T, B, N, D_out)


class DynamicAdjGenerator(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads)

    def forward(self, node_embs):
        """
        :param node_embs: (T, B, N, D)
        :return adj: (T, N, N)
        """
        T, B, D = node_embs.shape
        print(T, B, D)
        N = 1
        adj = torch.zeros(T, B, N, device=node_embs.device)

        for t in range(T):
            # 时间步t的节点嵌入
            emb_t = node_embs[t]  # (B, N, D)

            # 多头注意力计算相似度
            attn_output, _ = self.attention(
                emb_t.view(B * N, 1, D),
                emb_t.view(B * N, 1, D),
                emb_t.view(B * N, 1, D)
            )
            sim_matrix = torch.cosine_similarity(
                attn_output.view(B, N, D),
                attn_output.view(B, N, D).unsqueeze(2),
                dim=-1
            )  # (B, N, N)

            # 批次平均得到时间步t的adj
            print(sim_matrix.shape)
            adj[t] = sim_matrix.mean(dim=0)

        return adj  # (T, N, N)


class STFusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, diffusion_steps=2):
        super().__init__()
        self.adj_generator = DynamicAdjGenerator(hidden_dim, num_heads=4)
        self.st_block = SpatioTemporalBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps,
        )

    def forward(self, x_seq):
        """
        :param x_seq: 输入序列 (T, B, N, D_in)
        :return outputs: (T, B, N, D_out)
        """
        # 1. 生成动态邻接矩阵序列
        adj_seq = self.adj_generator(x_seq)  # (T, N, N)

        # 2. 时空融合处理
        outputs = self.st_block(x_seq, adj_seq)

        return outputs