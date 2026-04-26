import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.class_Crossformer import PatchEmbedding
from layers.class_Crossformer import AttentionLayer, FullAttention, TwoStageAttentionLayer
from layers.class_Crossformer import FlattenHead
import numpy as np


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.useweather=configs.useweather
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len#12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len  
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len

        self.in_seg_num = self.pad_in_len // self.seg_len # 能整除的分割
        # 第二层开始每层 除2 
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1))) # 向上取整
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding  他这个是不重叠的 patch ，就是分割 
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
            ]
        )
        # Decoder
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
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )
        # if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
        #                             head_dropout=configs.dropout)
        # elif self.task_name == 'classification':
        #     self.flatten = nn.Flatten(start_dim=-2)
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         self.head_nf * configs.enc_in, configs.num_class)


    def forecast(self, x_enc):
        # x : [B,L,C]
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        # 上一步执行了 dmodel 映射， patch操作， 加 position encoding 操作
        # 得到的 x ：  [B*C,pacth_number,d]
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars) # 把 channel 分出来
        x_enc += self.enc_pos_embedding # 加上一个可学习的张量，广播机制
        x_enc = self.pre_norm(x_enc) # layer norm 
        # 当前 x: [B， C,pacth_number,d]
        enc_out, attns = self.encoder(x_enc) # 进入 encoder层
        # 随后进入 decoder
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    # def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
    #     # embedding
    #     x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
    #     x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
    #     x_enc += self.enc_pos_embedding
    #     x_enc = self.pre_norm(x_enc)
    #     enc_out, attns = self.encoder(x_enc)

    #     dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

    #     return dec_out

    # def anomaly_detection(self, x_enc):
    #     # embedding
    #     x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
    #     x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
    #     x_enc += self.enc_pos_embedding
    #     x_enc = self.pre_norm(x_enc)
    #     enc_out, attns = self.encoder(x_enc)

    #     dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
    #     return dec_out

    # def classification(self, x_enc, x_mark_enc):
    #     # embedding
    #     x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

    #     x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
    #     x_enc += self.enc_pos_embedding
    #     x_enc = self.pre_norm(x_enc)
    #     enc_out, attns = self.encoder(x_enc)
    #     # Output from Non-stationary Transformer
    #     output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
    #     output = self.dropout(output)
    #     output = output.reshape(output.shape[0], -1)
    #     output = self.projection(output)
    #     return output

    def forward(self, x_enc, x_mark_enc,w_enc):
        
        # x=torch.cat([w_enc,x_enc],dim=-1)
        if self.useweather:
            x=torch.cat([w_enc,x_enc],dim=-1)
        else:
            x=x_enc
        # print(x.shape)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # if self.task_name == 'imputation':
        #     dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc)
        #     return dec_out  # [B, N]
        # return None