__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp




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
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        add_itransformer = configs.add_itransformer
        self.add_cross_channel_attention=configs.add_cross_channel_attention

        # self.channel_fusion = ChannelConvFusion(in_channels=2, out_channels=1)
        # self.channel_fusion = UNET_1D(input_dim= 2, num_features= 8, kernel_size=3)

        self.addmasknight=configs.addmasknight
        if configs.addmasknight:
            pass
            # self.A = nn.Parameter(torch.tensor(1.0))    # 振幅
            # self.sigma = nn.Parameter(torch.tensor(4.0))# 标准差，约束扩散程度
            # self.C = nn.Parameter(torch.tensor(0.1))     # 基线值
            # self.g_generator = GGenerator() 
        
        
        # ----------------------by yujia-----------------------#     
        self.change_embed=configs.change_embed # True or False
        self.change_embed_dim=configs.change_embed_dim
        self.use_head_embed=configs.use_head_embed
        self.n_head_embed=configs.n_head_embed
        self.mlp_embed_layer=configs.mlp_embed_layer
        self.dmodel_embed=configs.dmodel_embed
        self.useweather=configs.useweather

        if self.change_embed:
            # self.radiant_mlp=nn.Sequential(
            #     nn.Linear(672, 672-96),
            #     nn.ReLU(),
            #     # nn.Linear(672-96, 672-96),
            #     # nn.ReLU(),
            #     # nn.Linear(672-96, 672),
            #     # nn.ReLU(),            
            # )
            # self.fusion_fc=nn.Linear(672+96, 672)
            if self.use_head_embed==False:
                self.fusion_fc = Fusion_MLP(
                    input_dim=pred_len+self.change_embed_dim,
                    hidden_dim=2*(pred_len+self.change_embed_dim),
                    output_dim=pred_len,
                    num_layers=self.mlp_embed_layer,
                    use_head_embed=self.use_head_embed,
                    n_head_embed=self.n_head_embed
                )
            else:
                self.fusion_attention = Attention_fusion_yj(
                    d_model=self.dmodel_embed,
                    n_heads=self.n_head_embed,
                    use_attention=True,   
                    use_pos=True,
                    use_segment=True,
                    seq_len_x=self.change_embed_dim,
                    seq_len_w=configs.pred_len,
                    output_len=configs.pred_len
                )
        # ----------------------by yujia-----------------------#   

        # model
        self.decomposition = decomposition
        self.add_itransformer = add_itransformer
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose,change_embed=self.change_embed,change_embed_dim=self.change_embed_dim, **kwargs)
            if self.add_itransformer:
                self.itransformer = iTransformer.Model(configs).float()
    
    # 默认不进行composition
    def forward(self, x, w, w_mark):           # x: [Batch, Input length, Channel] w is weather prediction data
        # x=torch.cat([w,x],dim=-1)
        if self.useweather:
            x=torch.cat([w,x],dim=-1)
        else:
            x=x
        if self.decomposition:
            # print('inter PatchTST success','currerent x shap:',x.shape)
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1) 
             # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)   
            # x: [Batch, Channel, Input length]
        else:
            # print(x.shape)
            x = x.permute(0,2,1)    
            # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)   
             # x: [Batch, Output length, Channel]
            # print(x.shape)  # torch.Size([32, 672, 1])
            # x[:,:,-1]=torch.clamp(x[:,:,-1], min=0)
            #print(x.flatten(start_dim = 1, end_dim = 2)[:1, :].flatten())
            if self.add_itransformer:
                # 通道上 cat
                itransformer_input = torch.cat((w, x), 2)
                # print(f"itransformer_input shape: {itransformer_input.shape}")
                out = self.itransformer(itransformer_input, w_mark)
                return out
            # elif self.add_cross_channel_attention:
            #     x = self.channel_fusion(x)
            #     '''
            #     下面三行是负值归0操作
            #     '''
            #     # new_out = out.clone()
            #     # new_out[:,:,-1] = torch.clamp(out[:,:,-1], min=0)
            #     # x = new_out
            # 添加MLP-embed fusion操作
            if self.add_cross_channel_attention:
                x = x.permute(0,2,1)  # x: [Batch, Channel, Input length] 这里只有最后一个维度
                w= w.permute(0,2,1)   # x: [Batch, Channel, Input length] 这里只有最后一个维度
                radiant_embed=w # self.radiant_mlp(w)
                # print(x.shape)
                # print(w.shape)
                if self.use_head_embed==False:
                    fused= torch.cat([x, radiant_embed], dim=-1)  # torch.Size([32, 1, 768])
                    # print(fused.shape)
                    x=self.fusion_fc(fused)
                else:
                    x=self.fusion_attention(x,w)

                x=x.permute(0,2,1)  # x: [Batch, Input length，Channel]
                # 添加夜间mask操作
                if self.addmasknight:
                    new_out = x.clone()
                    new_out[:,:,-1] = torch.clamp(x[:,:,-1], min=0)
                    x = new_out
        return x

