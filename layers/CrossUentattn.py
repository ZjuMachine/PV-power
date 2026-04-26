import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils.masking import TriangularCausalMask, ProbMask
import math
from math import sqrt
from einops import rearrange, repeat
import numpy as np
from torch.func import vmap


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : [B,C，L]
        # do patching
        n_vars = x.shape[1]
        # padding
        x = self.padding_patch_layer(x)
        # unfold，along L，length is pacth len，stride
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # reshap x  ---> [B*C,pacth_number,patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        # project  dmodel plus position encoding --> [B*C,pacth_number,d]
        return self.dropout(x), n_vars



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # print(queries.shape, keys.shape, values.shape)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.space_attn = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.nvarjection=False
        self. swichchannel=configs.swichchannel
        self.nvars=configs.enc_in
        dffnow=4*self.nvars
        self.MLP_space1 = nn.Sequential(nn.Linear(self.nvars, dffnow),
                                  nn.GELU(),
                                  nn.Linear(dffnow, self.nvars))
        
        self.MLP_space2 = nn.Sequential(nn.Linear(self.nvars, dffnow),
                                  nn.GELU(),
                                  nn.Linear(dffnow, self.nvars))
        
        self.fc1ccc = nn.Sequential(nn.Linear(self.nvars, self.nvars//2),
                                  nn.GELU(),
                                  nn.Linear(self.nvars//2, self.nvars))
        
    def compute_single_corr(self,sample):
        sample_transposed = sample.T
        return torch.corrcoef(sample_transposed)
    def compute_channel_correlation(self, dec_out,corr):
        if self.swichchannel==True:
            dec_out_att = torch.bmm(dec_out.permute(0,2,1), corr).permute(0,2,1)     # [B, L, C]
        else:
            dec_out_att = torch.bmm(corr,dec_out)     # [B,C,L]
        return dec_out_att #.permute(0,2,1) # [B, L, C]
    def forward(self, x,corr, attn_mask=None, tau=None, delta=None):
        atten_new=[]
        # 选择attention
        use_pcorr=True
        # x：  [B， C, pacth_number,d]
        x_tmp=x.clone() #  [B， C, pacth_number,d]
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        B, nvars, pacth_number, d_model = x.shape 
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model') # 合并维度
        # # time attention
        time_enc, attn  = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        # time_enc:  (b ts_d) seg_num d_model
        x_new = rearrange(x_tmp, 'b ts_d seg_num d_model -> b ts_d (seg_num d_model)')
        corr_matrix = self.compute_channel_correlation(x_new,corr)  
        atten_new=[x_new.detach(),corr.detach(), corr_matrix.detach()] # 输入前， 相关性，输入后
        x_3= corr_matrix.view(B, nvars, pacth_number, d_model)
        time_reco=time_enc.view(B,nvars,pacth_number, d_model)
        dim_in=x+self.dropout(time_reco+x_3)
        dim_in = self.norm1(dim_in) #  -->(B,nvars,pacth_number, d_model)

        if self.nvarjection==False:
            dim_in = dim_in + self.dropout(self.MLP1(dim_in)) 
        else:
            dim_in_channel=dim_in.clone().permute(0,2,3,1) #  -->(B,pacth_number, d_model,nvars)
            dim_in_time=dim_in.clone()
            dim_in_channel=self.dropout(self.MLP_space1(dim_in_channel))
            dim_in_time=self.dropout(self.MLP1(dim_in))
            dim_in = dim_in + (dim_in_channel.permute(0,3,1,2)+dim_in_time)/2

        dim_in = self.norm2(dim_in)
        final_out = dim_in
        return final_out, atten_new

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # print(queries.shape, keys.shape, values.shape)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # print(queries.shape, keys.shape, values.shape)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
