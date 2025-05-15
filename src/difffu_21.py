import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from transformer import ns_model_1
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(args,schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = args.scale
        beta_start = scale * args.beta_start
        beta_end = scale * args.beta_end
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = args.scale
        beta_start = scale * args.beta_start + 0.01
        beta_end = scale * args.beta_end + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = args.scale
        beta_start = scale * args.beta_start + 0.01
        beta_mid = scale * args.beta_start * 10  #scale * 0.02
        beta_end = args.beta_end
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return  self.norm(x + self.dropout(sublayer(x)))
    
class SublayerConnection_in(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection_in, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer,idx):
        "Apply residual connection to any sublayer with the same size."
        # if idx == 0:
        #     return  x + self.dropout(sublayer(x))
        # else:
        return  self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))
    


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden

class MultiModalConditionEncoder(nn.Module):
    """多模态条件编码器"""
    def __init__(self, hidden_size, time_dim, ent_dim, rel_dim):
        super().__init__()
        # 各模态投影层
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_size//2),
            nn.GELU()
        )
        self.entity_proj = nn.Sequential(
            nn.Linear(ent_dim, hidden_size//2),
            nn.LayerNorm(hidden_size//2)
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(rel_dim, hidden_size),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.GLU(dim=-1)
        )

    def forward(self, cond_info):
        # 分模态编码
        time_feat = self.time_proj(cond_info['time'])  # [B, D/2]
        ent_feat = self.entity_proj(cond_info['object'])  # [B, D/2]
        rel_feat = self.relation_proj(cond_info['relation'])  # [B, D]

        # 时空融合
        time_ent = torch.cat([time_feat, ent_feat], dim=-1)  # [B, D]
        # 关系条件调制
        fused = self.fusion(torch.cat([time_ent, rel_feat], dim=-1))  # [B, D]

        return fused

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.condition_encoder = MultiModalConditionEncoder(
            hidden_size=hidden_size,
            time_dim=hidden_size,
            ent_dim=hidden_size,
            rel_dim=hidden_size
        )

        # 条件融合门控
        self.cond_gate = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Sigmoid()
        )
#        # 新增条件投影层
#        self.cond_projection = nn.Sequential(
#            nn.Linear(hidden_size, 4*hidden_size),
#            nn.GLU(dim=-1)  # 输出维度变为2*hidden_size
#        )
#        self.cond_gate = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, hidden, mask, cond_info):

#        hidden = self.attn_layer(hidden, cond_info)
#        hist_cond = self.hist_encoder(cond_info)
#        cond = self.cond_projection(hist_cond)  # [B, hidden_size*2]
#        gate = torch.sigmoid(self.cond_gate(cond))  # [B, hidden_size]
#        
#
#        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
#        
#        hidden = hidden * gate
#        hidden = self.output_sublayer(hidden, self.feed_forward)
#        return self.dropout(hidden)
        cond = self.condition_encoder(cond_info)  # [B, hidden_size]

        # 条件增强的自注意力
        hidden = self.input_sublayer(
            hidden,
#            lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask)
            lambda h: self.conditional_attention(h, cond, mask)
        )
        gate = self.cond_gate(torch.cat([hidden, cond], dim=-1))
        gated_hidden = hidden * gate

        # 条件增强的前馈
        output = self.output_sublayer(gated_hidden, self.feed_forward)
        
        return self.dropout(output)
        
    def conditional_attention(self, hidden, cond, mask):
        """条件注入的注意力计算"""
        # 将条件信息融入查询向量
        cond_query = hidden + cond  # [B, 1, D] → [B, L, D]
        return self.input_sublayer(cond_query, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
    
class Cross_TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(Cross_TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask,query_sub3=None,idx=0):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size 
        self.heads = args.heads
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.n_blocks_cross = args.num_blocks_cross
        self.dropout_hidden = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])
        self.linear_integrate = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size,self.hidden_size),
                nn.Sigmoid()
        )
        if self.n_blocks_cross !=0:
            self.cross_transformer_blocks = nn.ModuleList(
                [Cross_TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks_cross)])

    def forward(self, hidden, mask, cond_info):
        for idx,transformer in enumerate(self.transformer_blocks):
            hidden = transformer.forward(hidden, mask, cond_info)
        return hidden


class Transformer_rep_new(nn.Module):
    def __init__(self, config, args):
        super(Transformer_rep_new, self).__init__()
        self.config = config
        self.start_fc = nn.Linear(in_features=config.input_dim, out_features=config.channels)
        self.layers = nn.ModuleList()
        self.skip_generators = nn.ModuleList()
        self.hidden_size = args.hidden_size
        self.heads = args.heads
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.n_blocks_cross = args.num_blocks_cross
        self.dropout_hidden = nn.Dropout(0.4)
        cuts = config.lag
        for patch_size in config.patch_sizes:
            if cuts % patch_size != 0:
                raise Exception('Lag not divisible by patch size')
            cuts = int(cuts / patch_size)
            self.layers.append(Layer(device=config.device, input_dim=config.channels,
                                     dynamic=config.dynamic, num_nodes=config.num_nodes, cuts=cuts,
                                     cut_size=patch_size, factorized=True))
            self.skip_generators.append(WeightGenerator(in_dim=cuts * config.channels, out_dim=200, number_of_weights=1,
                                                        mem_dim=config.mem_dim, num_nodes=config.num_nodes, factorized=False))

        self.custom_linear = CustomLinear(factorized=False)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])
        self.linear_integrate = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size,self.hidden_size),
                nn.Sigmoid()
        )
        if config.num_blocks_cross != 0:
            self.cross_transformer_blocks = nn.ModuleList(
                [Cross_TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks_cross)])
        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.horizon)])

    def forward(self, batch_x, mask, query_sub3 = None):
        batch_size, seq_len, hidden_size = batch_x.shape
        batch_x = batch_x.view(batch_size, seq_len, 1, hidden_size)
        x = self.start_fc(batch_x)
        batch_size = x.size(0)
        skip = 0

        for layer, skip_generator in zip(self.layers, self.skip_generators):
            x = layer(x)
            weights, biases = skip_generator()
            skip_inp = x.transpose(2, 1).reshape(batch_size, 1, self.config.num_nodes, -1)
            skip = skip + self.custom_linear(skip_inp, weights[-1], biases[-1])
        hidden = torch.relu(skip).squeeze(1)

        for idx, transformer in enumerate(self.transformer_blocks):
            hidden = transformer.forward(hidden, mask, idx)

        return hidden


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.Transformers = Transformer_rep(args)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.after_diffu_rep = LayerNorm(self.hidden_size)
        self.before_diffu_rep = LayerNorm(self.hidden_size)
        self.pattern_noise_radio = args.pattern_noise_radio

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, cond_info, t,sr_embs, mask_seq,query_sub3=None):
    
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        
        laten_res = rep_item.clone()
        
        laten_res= self.before_diffu_rep(laten_res + emb_t.unsqueeze(1))

        rep_diffu = self.Transformers(laten_res, mask_seq, cond_info)

        out = [rep_diffu[:, -1, :],rep_diffu[:, 0, :]]
        
        return out




class DiffuRec(nn.Module):
    def __init__(self, args,):
        super(DiffuRec, self).__init__()
        self.args = args
        self.k_step = args.k_step
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.xstart_model = Diffu_xstart(self.hidden_size, args)
        

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(self.args,noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    
    
    def q_sample(self, x_start, t, cond_pred_model, embeddings_dict, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
            
        cond_noise = cond_pred_model(noise, embeddings_dict['object'] + embeddings_dict['relation'])
        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#            + (1 - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * cond_noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def q_sample_adj(self, x_start, t, noise=None, mask=None, adj_matrix=None):

        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # 噪声注入公式：x_t = sqrt(α_cumprod) * x_0 + sqrt(1-α_cumprod) * ε

        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        # 动态图引导的特征传播
        if adj_matrix is not None:
            x_t = torch.matmul(adj_matrix, x_t)  # 图卷积操作

        return x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    # def scale_t(self, ts):
    #     map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
    #     new_ts = map_tensor[ts]
    #     # print(new_ts)
    #     if self.rescale_timesteps:
    #         new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean
    
    def ddim_step(self, x_t, repres, t_now, t_next,e_embs):
        denoised_obj = torch.softmax(repres[:,-1,:] @ e_embs[:-2].t(),-1) @ e_embs[:-2]
        
        alpha_now = _extract_into_tensor(self.alphas_cumprod,t_now,denoised_obj.shape)
        noise_obj = torch.rsqrt(1 - alpha_now) * (x_t[:,-1,:] - torch.sqrt(alpha_now) * denoised_obj)
        

        return self.q_sample(denoised_obj,t_next,noise_obj)
    
    def p_sample_ddim(self, item_rep,noise_x_t, t,e_embs, mask_seq,mask=None):
        model_output, repres = self.xstart_model(item_rep, [], self._scale_timesteps(t),e_embs, mask_seq)
        x_0 = model_output
        if max(t) == 0:
            return x_0
        sample_xt = self.ddim_step(item_rep,repres,t,t-self.k_step,e_embs)
        return sample_xt
    
    def p_mean_variance(self, item_rep, cond_info, t,sr_embs, mask_seq,query_sub3=None):
        model_output = self.xstart_model(item_rep, cond_info, self._scale_timesteps(t),sr_embs, mask_seq,query_sub3)
        
        x_0 = model_output  ##output predict
        return x_0, []

    def p_sample(self, item_rep, cond_info, t,sr_embs, mask_seq,mask=None,query_sub3=None):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, cond_info, t,sr_embs, mask_seq,query_sub3)
        return model_mean

    def reverse_p_sample(self, item_rep, cond_info,sr_embs, mask_seq,mask=None,query_sub3=None):
        device = next(self.xstart_model.parameters()).device
        indices = list(range(0,self.num_timesteps))[::-1]
        if self.k_step == 0:
            for i in [max(indices)]: # from T to 0, reversion iteration  
                t = th.tensor([i] * item_rep.shape[0], device=device)
                with th.no_grad():
                    noise_x_t = self.p_sample(item_rep, cond_info, t,sr_embs, mask_seq,mask,query_sub3)
        return noise_x_t

    def forward(self, item_rep, cond_info, sr_embs, mask_seq,t,query_sub3=None):
        x_0, item_rep_out = self.xstart_model(item_rep, cond_info, self._scale_timesteps(t),sr_embs, mask_seq,query_sub3)  ##output predict

        return x_0, item_rep_out 
 

