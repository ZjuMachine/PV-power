import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.CrossUentattn import AttentionLayer, FullAttention,TwoStageAttentionLayer
from math import ceil

class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)
    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape  # B D segnum dmodel
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :]) 
        x = torch.cat(seg_to_merge, -1)
        # --> [batch_size, ts_d, seg_num // win_size, win_size * d_model]
        x = self.norm(x)
        x = self.linear_trans(x)
        # -->[batch_size, ts_d, seg_num // win_size, d_model]
        return x

class CNNMerging(nn.Module):
    def __init__(self, d_model, win_size,seg_num,norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=win_size,
            padding=1,
        )
        self.norm = norm_layer(seg_num * d_model)
    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        x= x.permute(0, 1, 3, 2).reshape(batch_size * ts_d, d_model, seg_num)
        x= self.conv(x)  # --> batch_size, ts_d, seg_num//2*d_model
        x = x.reshape(batch_size*ts_d, d_model*seg_num//2)
        #--> batch_size, ts_d, seg_num//2*d_model
        x=x.reshape(batch_size, ts_d, d_model,seg_num//2).permute(0, 1, 3, 2)
        return x

class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()
        self.use_conv=configs.convmerge
        if win_size > 1:
            if self.use_conv==False:
                self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
            else:
                self.merge_layer = CNNMerging(d_model, win_size,seg_num, nn.LayerNorm)
        else:
            self.merge_layer = None
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            if configs.twofilter:
                self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                                d_ff, dropout))
            else:
                self.encode_layers.append(ParellTwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                                d_ff, dropout))
                

    def forward(self, x,corr, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x, atten_new = layer(x,corr)
        return x, atten_new 

class Encoder(nn.Module):
    def __init__(self, attn_layers,configs,in_seg_num,win_size):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)
        final_seg_num = ceil(in_seg_num / win_size **(configs.e_layers - 1))
        self.bottleneck = BottleneckLayer(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            seg_num=final_seg_num,
            configs=configs
        )
    def forward(self, x,corr):
        #  x: [B， C, pacth_number, dmodel] 
        encode_x = []
        attenweights=[]
        encode_x.append(x)
        for block in self.encode_blocks:
            x, attns = block(x,corr)
            encode_x.append(x)
            attenweights.append(attns)
        bottleneck_out = self.bottleneck(x, corr)
        encode_x.append(bottleneck_out)
        return encode_x, attenweights #None

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)
    def forward(self, x, cross,corr):
        batch = x.shape[0]
        x,newattn = self.self_attention(x,corr)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict, newattn

class Decoder(nn.Module):
    def __init__(self, layers,configs):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)
        self.usebottle = configs.usebottle
    def forward(self, x, cross_all,corr):
        if self.usebottle==True:
            bottleneck_output = cross_all[-1]
            cross = cross_all[:-1]  
        else: 
            cross=cross_all
        final_predict = None
        i = 0
        layers_pre=[]
        attns_decs=[]
        layernum=len(self.decode_layers)
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            if self.usebottle:
                if i==0:
                    cross_enc = cross[layernum-i-1]
                    bottleneck_output=bottleneck_output+x[:,:,:bottleneck_output.size(2),:]
                    x, layer_predict, attns_decs_eachlayer = layer(x, bottleneck_output,corr)
                else:
                    cross_enc = cross[layernum-i-1]
                    x, layer_predict, attns_decs_eachlayer  = layer(x, cross_enc,corr)
            else:
                cross_enc = cross[i]
                x, layer_predict, attns_decs_eachlayer  = layer(x, cross_enc,corr)
            predappend= rearrange(layer_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
            layers_pre.append(predappend)
            attns_decs.append( attns_decs_eachlayer)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
        return final_predict,layers_pre,attns_decs

class BottleneckLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout,seg_num,configs):
        super(BottleneckLayer, self).__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, corr=None):
        # # Input shape: [batch_size, ts_d, seg_num, d_model]
        batch_size, ts_d, seg_num, d_model = x.shape
        # # Reshape for self-attention
        x_reshaped = x.reshape(batch_size * ts_d, seg_num, d_model)
        # # Apply self-attention
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = x_reshaped + self.dropout(attn_output)
        x_reshaped = self.norm1(x_reshaped)
        # # Apply feed-forward network
        ff_output = self.feed_forward(x_reshaped)
        x_reshaped = x_reshaped + self.dropout(ff_output)
        x_reshaped = self.norm2(x_reshaped)
        # # Reshape back to original dimensions
        x_output = x_reshaped.reshape(batch_size, ts_d, seg_num, d_model)
        return x_output
    

class ParellTwoStageAttentionLayer(nn.Module):
    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ParellTwoStageAttentionLayer, self).__init__()
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
        dec_out=dec_out.permute(0,2,1)
        # B, L, C = dec_out.shape
        dec_out_att = torch.bmm(dec_out, corr)     # [B, L, C]
        return dec_out_att.permute(0,2,1) # [B, L, C]
    def forward(self, x,corr, attn_mask=None, tau=None, delta=None):
        # x：  [B， C, pacth_number,d]
        B, nvars, pacth_number, d_model = x.shape 
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model') 
        # enter time attention
        time_enc, attn  = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        # time_enc:  (b ts_d) seg_num d_model
        #  reshape => [B, nvars, L, d_model] => permute => [B*L, nvars, d_model]
        x_2 = x.permute(0,2,1,3).contiguous().view(B*pacth_number, nvars, d_model)
        space_enc, attn_2  = self.time_attention(x_2, x_2, x_2, attn_mask=None, tau=None, delta=None)
        #  space_enc [B*seg_num , nvars, d_model]
        time_reco=time_enc.view(B,nvars,pacth_number, d_model)
        space_reco=space_enc.view(B,pacth_number,nvars, d_model).permute(0,2,1,3)
        atten_new=[time_reco.detach(), space_reco.detach()]
        dim_in=x+self.dropout(time_reco+space_reco)
        dim_in = self.norm1(dim_in) #  -->(B,nvars,pacth_number, d_model)
        if self.nvarjection==False:
            dim_in = dim_in + self.dropout(self.MLP1(dim_in)) #  -->(B,nvars,pacth_number, d_model)
        else:
            dim_in_channel=dim_in.clone().permute(0,2,3,1) #  -->(B,pacth_number, d_model,nvars)
            dim_in_time=dim_in.clone()
            dim_in_channel=self.dropout(self.MLP_space1(dim_in_channel))
            dim_in_time=self.dropout(self.MLP1(dim_in))
            dim_in = dim_in + (dim_in_channel.permute(0,3,1,2)+dim_in_time)/2
        dim_in = self.norm2(dim_in)
        final_out = dim_in
        return final_out, atten_new


























































































