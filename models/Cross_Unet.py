import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.func import vmap
from layers.CrossUnet_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.CrossUentattn import AttentionLayer, FullAttention, TwoStageAttentionLayer,PatchEmbedding
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len #12
        self.win_size = 2
        self.task_name = configs.task_name
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len  
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len 
        # divide 2 
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1))) # 向上取整
        self.head_nf = configs.d_model * self.out_seg_num
        # Embedding  
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)
        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ],
            configs, self.in_seg_num,self.win_size
        )
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))
        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for l in range(configs.e_layers + 1)
            ],configs
        )
        self.useweather=configs.useweather
        self.usenonlinearproject=configs.usenonlinearproject
        self.channelprohec=nn.Sequential(
            nn.Linear(self.enc_in  ,self.enc_in*4,False),
            nn.Sigmoid(),
            nn.Linear(self.enc_in*4,self.enc_in ,False),
        )
        self.channel_proj2=nn.Sequential(
            nn.Linear(self.enc_in*self.enc_in ,self.enc_in*4*self.enc_in,False),
            nn.Sigmoid(),
            nn.Linear(self.enc_in*4*self.enc_in,self.enc_in*self.enc_in ,False),
        )
    def forecast(self, x_enc,x_mark_dec,corr):
        x_tmp=x_enc.clone()
        # x : [B,L,C]
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        #  x ：  [B*C,pacth_number,d]
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars) 
        x_enc += self.enc_pos_embedding # position embedding
        x_enc = self.pre_norm(x_enc) #  x: [B， C,pacth_number,d]
        enc_out, attns = self.encoder(x_enc,corr)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out_final,dec_each_layer,attns_decs= self.decoder(dec_in, enc_out,corr)
        dec_out=dec_out_final
        return dec_out
    def compute_single_corr(self,sample):
        sample_transposed = sample.T
        return torch.corrcoef(sample_transposed)
    def compute_channel_correlation(self, dec_out):
        proj=self.usenonlinearproject
        B, L, C = dec_out.shape
        compute_batch_corr = vmap(self.compute_single_corr)
        corr = compute_batch_corr(dec_out)  # [C, C]
        # corr_reshaped = corr.view(B, -1)  # [B, C * C]
        corr_reshaped = corr[:,:-1,-1]  # [B, C ]
        if torch.any(torch.isnan(corr_reshaped)) or torch.any(torch.isinf(corr_reshaped)):
            corr_reshaped = torch.nan_to_num(corr_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
        mask = corr_reshaped < 0
        corr_reshaped[mask] = 0
        if proj==False:
            output_reshaped = F.softmax(corr_reshaped, dim=-1)
            output = output_reshaped.unsqueeze(-1)
            output = output.repeat(1,1,C-1) 
        else:
            output = corr_reshaped.unsqueeze(-1)
            output =self.channelprohec(output.permute(0,2,1)).permute(0,2,1)
            output = output.repeat(1,1,C-1) 
            output= output.view(-1,(C-1)*(C-1))
            output=self.channel_proj2(output)
            output_reshaped = F.softmax(output, dim=-1)
            output=output_reshaped.view(-1,C-1,C-1)
        return output # [B, L, C]

    def forward(self, x_enc, x_mark_enc,w_enc,x_mark_dec,seq_w_nwp_hist,seq_x_hist):
        newcat=torch.cat([seq_w_nwp_hist,seq_x_hist,x_enc[:,:,-1:]],dim=-1)
        corr=self.compute_channel_correlation(newcat)
        if self.useweather:
            x=torch.cat([w_enc,x_enc],dim=-1)
        else:
            x=x_enc
        dec_out = self.forecast(x,x_mark_dec,corr)
        return dec_out[:, :self.pred_len, :]  # [B, L, D]
