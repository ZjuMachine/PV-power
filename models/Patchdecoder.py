
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.cross_PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.yj_decoder import SegmentRestoration
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec_timeserieslab import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.class_Crossformer import AttentionLayer, FullAttention



class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        pred_len = configs.pred_len
        self.pred_len=pred_len
        kernel_size = configs.kernel_size
        self.adddecoder=configs.adddecoder
        self.change_embed_dim=96
        self.change_embed=False
        self.useweather=configs.useweather
        # model

        self.decomp_module = series_decomp(kernel_size)
        self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose,change_embed=self.change_embed,change_embed_dim=self.change_embed_dim, **kwargs)
        self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose,change_embed=self.change_embed,change_embed_dim=self.change_embed_dim, **kwargs)

        if self.adddecoder:
            self.dmodeluse= 128*4 # configs.d_model  #
            self.reseg=SegmentRestoration( d_model,patch_len)
            self.lineproject=nn.Linear(c_in,self.dmodeluse)
            self.dec_embedding = DataEmbedding(c_in, self.dmodeluse, configs.embed, configs.freq,
                                    configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            self.dmodeluse, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            self.dmodeluse, configs.n_heads),
                        self.dmodeluse,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.dmodeluse),
                projection=nn.Linear(self.dmodeluse, configs.c_out, bias=True)
            )
        else:
            pass
        self.fusion_fc=nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,6),
            nn.ReLU(),            
        )

        self.fusion_patches=nn.Sequential(
            nn.Linear(configs.seq_len*4*2, self.pred_len),
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len),
        )
    def forward(self, x,x_mark,x_dec, x_mark_dec,w):           # x: [Batch, Input length, Channel] w is weather prediction data
        if self.useweather:
            x=torch.cat([w,x],dim=-1)
        else:
            pass
        # x=torch.cat([w,x],dim=-1)
        x_temp=x
        res_init, trend_init = self.decomp_module(x)
        res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res = self.model_res(res_init,x_mark) #  [batch_size, C, L]
        trend = self.model_trend(trend_init,x_mark) #  [batch_size, C, L]            
        x_encout = res + trend #  [batch_size, C, L]  
        if self.adddecoder:
            enc_out = self.lineproject((x_encout.permute(0,2,1))) # -->[batch_size, L , dmodel]  
            # -->[batch_size, L , C]  
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
            x=dec_out[:,-self.pred_len:,:]
        else:
            radiant_embed=self.radiant_mlp(w.permute(0,2,1) ) # self.radiant_mlp(w)
            x_expanded = x_encout.expand(-1, w.size(2), -1) 
            fused= torch.cat([x_expanded, radiant_embed], dim=-1)
            x=self.fusion_fc(fused.permute(0,2,1))  #  [B L C]
        return x