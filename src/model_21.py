import torch.nn as nn
import torch
from torch import optim
import math
from difffu_21 import DiffuRec
import torch.nn.functional as F
import numpy as np
from regcn import RGCNCell,RGCNBlockLayer,AttEnhancedRGCNCell,AttEnhancedRGCNBlockLayer
import torch as th
from entmax import sparsemax, entmax15, entmax_bisect
from dcgru import STFusionModel
from Triformer import Layer
from transformer import ns_model_1
from torch_scatter import scatter
from utils import utils

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.weight_mlp = nn.Linear(hidden_size,hidden_size)
        self.bias_mlp = nn.Linear(hidden_size,hidden_size)
        self.variance_epsilon = eps

    def forward(self, x,weight=None):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if weight!=None:
            return self.weight_mlp(weight) * x + self.bias_mlp(weight)
        return self.weight * x + self.bias


class CurriculumScheduleSampler:
    def __init__(self, total_steps=50, max_t=1000):
        self.total_steps = total_steps
        self.max_t = max_t
        self.current_step = 0  # 训练步数计数器

    def get_stage_bounds(self):
        """动态划分噪声阶段"""
        if self.current_step < self.total_steps * 0.3:  # 前30%步数：低噪声
            return (0, int(self.max_t * 0.3))
        elif self.current_step < self.total_steps * 0.7:  # 30-70%步数：中噪声
            return (int(self.max_t * 0.3), int(self.max_t * 0.7))
        else:  # 后30%步数：全噪声
            return (int(self.max_t * 0.7), self.max_t)

    def sample(self, batch_size, device):
        t_min, t_max = self.get_stage_bounds()
        t = torch.randint(t_min, t_max, (batch_size,), device=device)
        self.current_step += 1  # 更新训练步数
        return t

class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args, encoder_name, num_ents, num_rels,
                num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False,
                 use_cuda=False,max_time=-1,
                 num_words=None,
                 num_static_rels=None):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size

        self.embed_dropout = nn.Dropout(args.dropout)
        self.time_max_len = max_time
        self.time_embeddings = nn.Embedding(self.time_max_len+1+1+1, self.emb_dim) # 1 for padding object and 1 for condition subject and 1 for cls
        # self.pos_embeddings = nn.Embedding(self.max_len+2, self.emb_dim)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm_static = LayerNorm(args.hidden_size, eps=1e-12)

        self.seen_label_embedding = nn.Embedding(2, self.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.max_len = args.max_len
        
        self.use_static = args.add_static
        self.layer_norm_gcn = args.layer_norm_gcn
        self.temperature_object = args.temperature_object
        self.pattern_noise_radio = args.pattern_noise_radio
        self.gpu = args.gpu
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.concat_con = args.concat_con
        self.refinements_radio = args.refinements_radio
        self.add_memory = args.add_memory
        self.add_ood_loss_energe = args.add_ood_loss_energe
        self.add_info_nce_loss = args.add_info_nce_loss

        self.seen_addition = args.seen_addition
        self.kl_interst = args.kl_interst
        self.add_frequence = args.add_frequence
        
        self.emb_rel = torch.nn.Parameter(torch.Tensor(num_rels*2, self.emb_dim),
                                          requires_grad=True).float()
        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, self.emb_dim),
                                              requires_grad=True).float()
        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(num_words, self.emb_dim), requires_grad=True).float()
            torch.nn.init.trunc_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.emb_dim, self.emb_dim, num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            
        self.gate_weight = nn.Parameter(torch.Tensor(1, self.emb_dim))
        nn.init.constant_(self.gate_weight,1)
        self.relation_cell_1 = nn.GRUCell(self.emb_dim*2, self.emb_dim)

        
        self.logistic_regression = nn.Linear(1,2)
        self.frequence_linear = nn.Linear(1,self.emb_dim,bias=False)
        
        self.mlp_model = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim*2),nn.GELU(),nn.Linear(self.emb_dim*2, 2))
        
        self.weight_energy = torch.nn.Parameter(torch.Tensor(1, self.num_ents),
                                              requires_grad=True).float()
        torch.nn.init.uniform_(self.weight_energy)
        torch.nn.init.trunc_normal_(self.emb_rel)
        torch.nn.init.trunc_normal_(self.emb_ent)
        torch.nn.init.trunc_normal_(self.time_embeddings.weight)

        torch.nn.init.uniform_(self.frequence_linear.weight,0,1)

        self.time2vec_freq = nn.Parameter(torch.randn(self.emb_dim/2))

    def _compute_sampling_threshold(self, batches_seen):

        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

    def denoise_step(self, x_t, t, labels):
        # 生成预测结果
        pred = self.model(x_t, t)

        # 课程学习混合
        cl_ratio = self._compute_sampling_threshold(self.batches_seen)
        mixed = cl_ratio * labels + (1 - cl_ratio) * pred

        return mixed

    def diffu_pre(self, item_rep, tag_emb, sr_embs, mask_seq,t,query_sub3=None):
        seq_rep_diffu, item_rep_out  = self.diffu(item_rep, tag_emb,sr_embs, mask_seq,t,query_sub3)
        return seq_rep_diffu, item_rep_out 

    def reverse(self, item_rep, noise_x_t,sr_embs, mask_seq,mask = None,query_sub3=None):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t,sr_embs, mask_seq,mask,query_sub3)
        return reverse_pre


    def loss_diffu_ce(self, rep_diffu, labels,history_tail_seq=None, one_hot_tail_seq=None,true_triples=None,mask_seq=None,emb_ent=None):
        
        

        loss = 0
        scores = (rep_diffu) @ emb_ent[:].t()/(math.sqrt(self.emb_dim)*self.temperature_object)
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        # """

        if self.add_ood_loss_energe:
            scores_c = self.mlp_model(rep_diffu)
            seen_before_label = one_hot_tail_seq.gather(1,labels.unsqueeze(-1))
            seen_entity = scores_c[seen_before_label.bool().squeeze(1)]
            unseen_entity = scores_c[~(seen_before_label.bool().squeeze(1))]
            input_for_lr = torch.cat((seen_entity, unseen_entity), 0)
            labels_for_lr = torch.cat((torch.ones(len(seen_entity)).cuda(), torch.zeros(len(unseen_entity)).cuda()), -1)
            lr_reg_loss = F.cross_entropy(input_for_lr/0.5, labels_for_lr.long())
            loss = loss +  lr_reg_loss
            
        return self.loss_ce(scores, labels.squeeze(-1)) + loss
    
    def regularization_memory(self):
        cos_mat = torch.matmul(self.emb_ent[:], self.emb_ent[:-2].transpose(1, 0))
        cos_sim = torch.norm((cos_mat - torch.eye(self.num_ents,self.num_ents).to(cos_mat.device))/math.sqrt(self.emb_dim))  ## not real mean
        return cos_sim
    
    def diffu_rep_pre(self, rep_diffu,e_embs,history_tail_seq=None, one_hot_tail_seq=None):
        
        scores = (rep_diffu[0]) @ e_embs[:-1].t()/(math.sqrt(self.emb_dim)*self.temperature_object)
        if history_tail_seq != None and self.seen_addition:
            history_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float)).to(self.gpu) * (- self.pattern_noise_radio)
            temp_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float) * (-1e9)).to(self.gpu)  # 等于0变成-1e9，等于1变成0
            history_frequency = F.softmax(history_tail_seq + temp_mask, dim=1) * self.refinements_radio  # (batch_size, output_dim)
            scores = history_mask + scores + history_frequency
        return scores


    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def _time2vec(self, t):
        """ Time2Vec周期性编码 """
       
        # 将时间差转换为周期性特征
        # 公式: [sin(α*t + β), cos(α*t + β), ...]
        t = t.unsqueeze(-1).float()  # (batch, seq_len) -> (batch, seq_len, 1)
        freq = self.time2vec_freq.abs()  # 确保频率为正
        sin_components = torch.sin(freq * t)  # (batch, seq_len, emb_dim)
        cos_components = torch.cos(freq * t)
        return torch.cat([sin_components, cos_components], dim=-1)  # (batch, seq_len, 2*emb_dim)

    
    def forward(self, glist, sequence, tag, train_f=True,use_cuda=True,y_0_hat_batch = None,seen_before_label=None,static_graph=None): 
        
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.emb_ent, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = self.LayerNorm_static(static_emb)
            padding = torch.zeros_like(static_emb[[-1]]).to(self.gpu)
            initial_h = torch.concat([static_emb,padding],dim=0)
        else:
            initial_h = self.emb_ent
            padding = torch.zeros_like(self.emb_ent[[-1]]).to(self.gpu)
            initial_h = torch.concat([initial_h,padding],dim=0)
            static_emb = None

        object_sequence = sequence[:,-self.max_len:,0] ### 获取历史交互的object序列
        relation_sequence = sequence[:,-self.max_len:,1] ### 获取历史发生的relation序列
        max_time = torch.max(sequence[:,-self.max_len:,2],dim=1,keepdim=True)[0] ### 获取以上序列对应的 time 序列
        time_sequence = max_time - sequence[:,-self.max_len:,2] #+ 1 + 1 ### 获取 relative time + 1 序列
        
        time_sequence = torch.concat([time_sequence,torch.ones_like(tag[:,[1]])],dim=-1)
        
        object_embeddings = initial_h[object_sequence].clone()
        relation_embeddings = self.emb_rel[relation_sequence].clone()
        
        query_rel3 = self.emb_rel[tag[:,1]].unsqueeze(1)

#        time_embeddings_re = self._time2vec(time_diff)
        time_embeddings_re = self.time_embeddings.weight[time_sequence] ## 获取对应的 embedding 序列
        mask_seq = (time_sequence<=max_time+2).float()  ## mask padding part as 0 # 生成掩码：有效时间步为1，padding为0


        relation_embeddings = torch.concat([relation_embeddings,query_rel3],dim=1)

        if train_flag:
            query_object3 = initial_h[tag[:,2]].unsqueeze(1)
            t, weights = self.diffu.schedule_sampler.sample(query_object3.shape[0], query_object3.device) ## t is sampled from schedule_sampler
            query_object_noise = self.diffu.q_sample(query_object3, t)
            
            object_embeddings = torch.concat([object_embeddings,query_object_noise],dim=1)
            
            inter_embeddings = object_embeddings+time_embeddings_re+relation_embeddings
            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(inter_embeddings)) ## dropout first than layernorm
            
            rep_diffu = self.diffu_pre(inter_embeddings_drop, inter_embeddings_drop[:,-1,:],initial_h, mask_seq,t)
            rep_item,object_protype_gt = None,None
        else:
            noise_x_t = th.randn_like(object_embeddings[:,[0,1,-1],:])
            
            query_object3 = noise_x_t[:,[2],:] 
            object_embeddings = torch.concat([object_embeddings,query_object3],dim=1)
            inter_embeddings = object_embeddings+time_embeddings_re + relation_embeddings
            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(inter_embeddings)) ## dropout first than layernorm

            rep_diffu = self.reverse(inter_embeddings_drop, inter_embeddings_drop[:,-1,:],initial_h, mask_seq,[])

            rep_item, t, object_protype_gt= None, None, None

        scores = None

        return scores, rep_diffu[0], rep_item, t, mask_seq,initial_h


    
class EnhancedAttDiffuseModel(nn.Module):
    def __init__(self, diffu, args, encoder_name, num_ents, num_rels,
                 num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False,
                 use_cuda=False, max_time=-1,
                 num_words=None,
                 num_static_rels=None):
        super(EnhancedAttDiffuseModel, self).__init__()
        self.emb_dim = args.hidden_size

        self.embed_dropout = nn.Dropout(args.dropout)
        self.pred_len = args.pred_len 
        self.label_len = args.label_len
        self.time_max_len = max_time
        
        self.gate_fc = nn.Sequential(
           nn.Linear(3 * self.emb_dim, self.emb_dim),
           nn.Sigmoid()
         )
         
        self.time_embeddings = nn.Embedding(self.time_max_len + 1 + 1 + 1,
                                            self.emb_dim)  # 1 for padding object and 1 for condition subject and 1 for cls
        # self.pos_embeddings = nn.Embedding(self.max_len+2, self.emb_dim)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm_static = LayerNorm(args.hidden_size, eps=1e-12)
        self.noise_alpha = nn.Parameter(torch.tensor(1.0))
        self.seen_label_embedding = nn.Embedding(2, self.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.max_len = args.max_len

        self.use_static = args.add_static
        self.layer_norm_gcn = args.layer_norm_gcn
        self.temperature_object = args.temperature_object
        self.pattern_noise_radio = args.pattern_noise_radio
        self.gpu = args.gpu
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.concat_con = args.concat_con
        self.refinements_radio = args.refinements_radio
        self.add_memory = args.add_memory
        self.add_ood_loss_energe = args.add_ood_loss_energe
        self.add_info_nce_loss = args.add_info_nce_loss
        self.rescale_timesteps = args.rescale_timesteps
        self.seen_addition = args.seen_addition
        self.kl_interst = args.kl_interst
        self.add_frequence = args.add_frequence
    
        self.emb_rel = torch.nn.Parameter(torch.Tensor(num_rels * 2, self.emb_dim),
                                          requires_grad=True).float()
        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, self.emb_dim),
                                          requires_grad=True).float()
        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(num_words, self.emb_dim), requires_grad=True).float()
            torch.nn.init.trunc_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.emb_dim, self.emb_dim, num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
    
        self.gate_weight = nn.Parameter(torch.Tensor(1, self.emb_dim))
        nn.init.constant_(self.gate_weight, 1)
        self.relation_cell_1 = nn.GRUCell(self.emb_dim * 2, self.emb_dim)
    
        self.logistic_regression = nn.Linear(1, 2)
        self.frequence_linear = nn.Linear(1, self.emb_dim, bias=False)
    
        self.mlp_model = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim * 2), nn.GELU(), nn.Linear(self.emb_dim * 2, 2))
    
        self.weight_energy = torch.nn.Parameter(torch.Tensor(1, self.num_ents),
                                                requires_grad=True).float()
        torch.nn.init.uniform_(self.weight_energy)
        torch.nn.init.trunc_normal_(self.emb_rel)
        torch.nn.init.trunc_normal_(self.emb_ent)
        torch.nn.init.trunc_normal_(self.time_embeddings.weight)
#        self.fusion_fc = nn.Linear(3 * self.emb_dim, 3)
        torch.nn.init.uniform_(self.frequence_linear.weight, 0, 1)
#        self.gru_memory = nn.GRU(self.emb_dim, self.emb_dim, num_layers=1)
        self.cl_decay_steps = 100

        self.cond_pred_model = ns_model_1.CrossAttentionFusion(noise_dim=self.emb_dim, history_dim=self.emb_dim)
        
        self.gate = ns_model_1.GateUnitExpand(factor_size= 3*self.emb_dim, hidden_size= self.emb_dim)

        self.Ws_attn = nn.Linear(self.emb_dim, 5, bias=False)
        self.Wr_attn = nn.Linear(self.emb_dim, 5, bias=False)
        self.Wqr_attn = nn.Linear(self.emb_dim, 5)
        self.w_alpha = nn.Linear(5, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU()

        self.time2vec_freq = nn.Parameter(torch.randn(int(self.emb_dim/2)))

#        self.cond_pred_model = ns_Transformer.Model(args).float().to(self.device)
#        self.cond_pred_model = nn.DataParallel(self.cond_pred_model, device_ids=[3])
    
    
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
    
    
    def denoise_step(self, x_t, t, labels):
        # 生成预测结果
        pred = self.model(x_t, t)
    
        # 课程学习混合
        cl_ratio = self._compute_sampling_threshold(self.batches_seen)
        mixed = cl_ratio * labels + (1 - cl_ratio) * pred
    
        return mixed
    
    
    def diffu_pre(self, item_rep, cond_info , sr_embs, mask_seq, t, query_sub3=None):
        seq_rep_diffu, item_rep_out = self.diffu(item_rep, cond_info , sr_embs, mask_seq, t, query_sub3)
        return seq_rep_diffu, item_rep_out
    
    
    def reverse(self, item_rep, cond_info, sr_embs, mask_seq, mask=None, query_sub3=None):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, cond_info, sr_embs, mask_seq, mask, query_sub3)
        return reverse_pre
    
    
    def loss_diffu_ce(self, rep_diffu, labels, history_tail_seq=None, one_hot_tail_seq=None, true_triples=None,
                      mask_seq=None, emb_ent=None):
        loss = 0
        scores = (rep_diffu) @ emb_ent[:].t() / (math.sqrt(self.emb_dim) * self.temperature_object)
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        # """
    
        if self.add_ood_loss_energe:
            scores_c = self.mlp_model(rep_diffu)
            seen_before_label = one_hot_tail_seq.gather(1, labels.unsqueeze(-1))
            seen_entity = scores_c[seen_before_label.bool().squeeze(1)]
            unseen_entity = scores_c[~(seen_before_label.bool().squeeze(1))]
            input_for_lr = torch.cat((seen_entity, unseen_entity), 0)
            labels_for_lr = torch.cat((torch.ones(len(seen_entity)).cuda(), torch.zeros(len(unseen_entity)).cuda()), -1)
            lr_reg_loss = F.cross_entropy(input_for_lr / 0.5, labels_for_lr.long())
            loss = loss + lr_reg_loss

#        adv_noise = torch.zeros_like(rep_diffu).normal_(0, 1e-5)
#        adv_loss = self.loss_ce((rep_diffu + adv_noise) @ emb_ent[:].t(), labels)+ 0.3 * adv_loss
        return self.loss_ce(scores, labels.squeeze(-1)) + loss
    
    
    def regularization_memory(self):
        cos_mat = torch.matmul(self.emb_ent[:], self.emb_ent[:-2].transpose(1, 0))
        cos_sim = torch.norm((cos_mat - torch.eye(self.num_ents, self.num_ents).to(cos_mat.device)) / math.sqrt(
            self.emb_dim))  ## not real mean
        return cos_sim
    
    
    def diffu_rep_pre(self, rep_diffu, e_embs, history_tail_seq=None, one_hot_tail_seq=None):
        scores = (rep_diffu[0]) @ e_embs[:-1].t() / (math.sqrt(self.emb_dim) * self.temperature_object)
        if history_tail_seq != None and self.seen_addition:
            history_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float)).to(self.gpu) * (
                - self.pattern_noise_radio)
            temp_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float) * (-1e9)).to(
                self.gpu)  # 等于0变成-1e9，等于1变成0
            history_frequency = F.softmax(history_tail_seq + temp_mask,
                                          dim=1) * self.refinements_radio  # (batch_size, output_dim)
            scores = history_mask + scores + history_frequency
        return scores
    
    
    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep / seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def _time2vec(self, t):
        """ Time2Vec周期性编码 """
       
        # 将时间差转换为周期性特征
        # 公式: [sin(α*t + β), cos(α*t + β), ...]
        t = t.unsqueeze(-1).float()  # (batch, seq_len) -> (batch, seq_len, 1)
        freq = self.time2vec_freq.abs()  # 确保频率为正
        sin_components = torch.sin(freq * t)  # (batch, seq_len, emb_dim)
        cos_components = torch.cos(freq * t)
        return torch.cat([sin_components, cos_components], dim=-1)  # (batch, seq_len, 2*emb_dim)
    
    
    def forward(self, glist, sequence, tag, epoch, entity_dict, rel_dict, train_flag=True, use_cuda=True, seen_before_label=None,
                static_graph=None):
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.emb_ent, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = self.LayerNorm_static(static_emb)
            padding = torch.zeros_like(static_emb[[-1]]).to(self.gpu)
            initial_h = torch.concat([static_emb, padding], dim=0)
        else:
            initial_h = self.emb_ent
            padding = torch.zeros_like(self.emb_ent[[-1]]).to(self.gpu)
            initial_h = torch.concat([initial_h, padding], dim=0)
            static_emb = None
#        model_optim = optim.Adam(self.cond_pred_model.parameters(), lr=0.0001)
#        model_optim.zero_grad()
#        sequence = sequence[:,-self.max_len:,:]
#        sequence, adjusted_sim = utils.select_relevant_slices_v1(tag, sequence, entity_dict, rel_dict, 128)
#        exit()
        sequence = sequence[:, -128:, :]
        object_sequence = sequence[:, :, 0]  ### 获取历史交互的object序列
        relation_sequence = sequence[:, :, 1]  ### 获取历史发生的relation序列
#        tail_sequence = sequence[:, :, 2]
        max_time = torch.max(sequence[:, :, 2], dim=1, keepdim=True)[0]  ### 获取以上序列对应的 time 序列
        time_sequence = max_time - sequence[:, :, 2] + 1 # + 1  ### 获取 relative time + 1 序列
#        print(time_sequence)
#        print(max_time)
#        print(self.time_embeddings.shape)
#        object_sequence = sequence[:,-self.max_len:,0] ### 获取历史交互的object序列
#        relation_sequence = sequence[:,-self.max_len:,1] ### 获取历史发生的relation序列
#        max_time = torch.max(sequence[:,-self.max_len:,2],dim=1,keepdim=True)[0] ### 获取以上序列对应的 time 序列
#        time_sequence = max_time - sequence[:,-self.max_len:,2] #+ 1 + 1 ### 获取 relative time + 1 序列
        
        
        object_embeddings = initial_h[object_sequence].clone()
#        print(relation_sequence.shape)
#        print(torch.max(relation_sequence))
        relation_embeddings = self.emb_rel[relation_sequence].clone()
#        print(time_sequence)
#        print(time_sequence.shape)
#        print(object_embeddings.shape)
#        exit()
        embeddings_dict = {
		    'object': object_embeddings.clone(),
		    'relation':  relation_embeddings.clone()
#		    'time': self._time2vec(time_sequence)
		}
        
#        print(time_sequence.shape)
#        print(tag.shape)
        time_sequence = torch.concat([time_sequence, torch.ones_like(tag[:, [1]])], dim=-1)
    
        query_rel3 = self.emb_rel[tag[:, 1]].unsqueeze(1)
#        print(time_sequence)
#        print(max_time)
#        exit()
#        time_embeddings_re = self._time2vec(time_sequence)
        time_embeddings_re = self.time_embeddings.weight[time_sequence]  ## 获取对应的 embedding 序列
        mask_seq = (time_sequence <= max_time).float()  ## mask padding part as 0 # 生成掩码：有效时间步为1，padding为0
    
        relation_embeddings = torch.concat([relation_embeddings, query_rel3], dim=1)
    
        if train_flag:

#            print(tag.shape)
#            print(initial_h.shape)
#            exit()
            
            query_object3 = initial_h[tag[:,2]].unsqueeze(1)
            
#            print(config.shape)
#            exit()
#            y_0_hat_batch = cond_pred_model(embeddings_dict['object'] + embeddings_dict['relation'] + embeddings_dict['time'])
#            print(y_0_hat_batch.shape)
#            exit()
            t, weights = self.diffu.schedule_sampler.sample(query_object3.shape[0],
                                                            query_object3.device)  ## t is sampled from schedule_sampler
#            y_0_hat_batch = initial_h[y_0_hat_batch[:,2]].unsqueeze(1)

#            temp = y_0_hat_batch[:,2].to(initial_h.device)
#            print(temp)
#            print(temp.shape)
#            print(tag.shape)
#            print(tag[:,2].shape)
#            print(query_object3.shape)
#            print(object_embeddings.shape)
#            print(object_sequence.shape)
#            exit()
			
		   
#            y_0_hat_batch = y_0_hat_batch.to(query_object3.device)
            query_object_noise = self.diffu.q_sample(query_object3, t , self.cond_pred_model, embeddings_dict)

            KL_loss = 0


    
            object_embeddings = torch.concat([object_embeddings, query_object_noise], dim=1)
#修改融合部分
#            combined = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)
#            inter_embeddings =self.fusion_fc(combined)
# 修改融合部分
#            combined = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)
#            gate = self.gate_fc(combined)
#            inter_embeddings = gate * object_embeddings + (1 - gate) * (time_embeddings_re + relation_embeddings)
            cond_info = {
		    'object': object_embeddings.clone(),
		    'relation': relation_embeddings.clone(),
		    'time': time_embeddings_re.clone()
		   }
#            inter_embeddings = self.gate(object_embeddings, time_embeddings_re, relation_embeddings, adjusted_sim)
#            num_ents = inter_embeddings.shape[0]
            
#            alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(object_embeddings) + self.Wr_attn(relation_embeddings) + self.Wqr_attn(time_embeddings_re)))
#            sigmoid_attention = torch.sigmoid(alpha)
#            inter_embeddings = sigmoid_attention * inter_embeddings 
            inter_embeddings = object_embeddings+time_embeddings_re+relation_embeddings
            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(inter_embeddings))  ## dropout first than layernorm
          
            rep_diffu = self.diffu_pre(inter_embeddings_drop, cond_info, initial_h, mask_seq, t)
            rep_item, object_protype_gt = None, None
        else:
            noise_x_t = th.randn_like(object_embeddings[:, [0, 1, -1], :])
    
            query_object3 = noise_x_t[:, [2], :]
            object_embeddings = torch.concat([object_embeddings, query_object3], dim=1)
            
            
#            combined = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)
#            inter_embeddings =self.fusion_fc(combined)
            
#            combined = torch.cat([object_embeddings, time_embeddings_re, relation_embeddings], dim=-1)
#            gate = self.gate_fc(combined)
#            inter_embeddings = gate * object_embeddings + (1 - gate) * (time_embeddings_re + relation_embeddings)

#            inter_embeddings = self.gate(object_embeddings, time_embeddings_re, relation_embeddings, adjusted_sim)
            inter_embeddings = object_embeddings+time_embeddings_re+relation_embeddings
##            num_ents = inter_embeddings.shape[0]
#            
#            alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(object_embeddings) + self.Wr_attn(relation_embeddings) + self.Wqr_attn(time_embeddings_re)))
#            sigmoid_attention = torch.sigmoid(alpha)
#            
#            inter_embeddings = sigmoid_attention * inter_embeddings
            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(inter_embeddings))  ## dropout first than layernorm
            cond_info = {
		    'object': object_embeddings.clone(),
		    'relation': relation_embeddings.clone(),
		    'time': time_embeddings_re.clone()
		   }
            
            rep_diffu = self.reverse(inter_embeddings_drop, cond_info, initial_h, mask_seq, [])
    
            rep_item, t, object_protype_gt = None, None, None
            KL_loss = 0
    
        scores = None
    
        return scores, rep_diffu[0], rep_item, t, mask_seq, initial_h, KL_loss
        
def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre




 
