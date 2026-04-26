import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.class_Crossformer import TwoStageAttentionLayer
from math import ceil


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        # B D segnum dmodel
        # win size是对于seg_num 进行降采样，就是隔几个取一个样本的问题

        batch_size, ts_d, seg_num, d_model = x.shape
        # 
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)


        # 他是在切片的数量维度上，隔一个取一个，稀疏拼接，然后再降采样到dmodel
        seg_to_merge = []
        for i in range(self.win_size):
            # 按照奇偶取出来
            seg_to_merge.append(x[:, :, i::self.win_size, :]) 
            
        x = torch.cat(seg_to_merge, -1)
        # 相邻拼接
        # 上面得到的是 --> [batch_size, ts_d, seg_num // win_size, win_size * d_model]

        x = self.norm(x)
        x = self.linear_trans(x)
        # 最终得到 -->[batch_size, ts_d, seg_num // win_size, d_model]
        # 上面的操作应该是在降采样
        return x


# 替换后的降采样层

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

        # Lout=(L+2p-3)/win_size+1  -> int((L//2-1)*win_size/2+3/2-L/2)=2p
        # L (1-win_size//2)-3 = 3-2p-win_size
    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape

        x= x.permute(0, 1, 3, 2).reshape(batch_size * ts_d, d_model, seg_num)

        # 在seg 上直接展平到时序维度，再做cnn做降采样
        # x = x.reshape(batch_size, ts_d, seg_num * d_model)
        #  
        x= self.conv(x)  # --> batch_size, ts_d, seg_num//2*d_model

        x = x.reshape(batch_size*ts_d, d_model*seg_num//2)
        # 
        # x=self.norm(x) # --> batch_size, ts_d, seg_num//2*d_model

        x=x.reshape(batch_size, ts_d, d_model,seg_num//2).permute(0, 1, 3, 2)

        return x


class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()
        # 第一层是 1 其他层是我们自己的 win size，是大于1的
        # 也就是除了第一层 都进行 merge
        self.use_conv=False # 这里试过了不行CNNMerging
        if win_size > 1:
            # 是否直接采用 conv 降采样 
            if self.use_conv==False:
                self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
            else:
                self.merge_layer = CNNMerging(d_model, win_size,seg_num, nn.LayerNorm)
    
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        self.seg_num=seg_num # 不断变化的



        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape

        # 
        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)
        
        # x=self.head(x)

        return x, None


class Encoder(nn.Module):
    def __init__(self, attn_layers,configs,seg_nums_list,win_size):
        super(Encoder, self).__init__()

        # 这里面就是 scale块
        self.encode_blocks = nn.ModuleList(attn_layers)

        self.seg_nums_list=seg_nums_list

        self.heads = nn.ModuleList()
        for seg_num in seg_nums_list:
            head = Flatten_Head(
                individual=True,
                n_vars=configs.enc_in,
                nf=configs.d_model * seg_num,  # 使用对应的segment number
                target_window=configs.pred_len,
                head_dropout=configs.fc_dropout
            )
            self.heads.append(head)
        # self.head=Flatten_Head(individual=True, 
        #                        n_vars=configs.enc_in, 
        #                        nf=d_model*self.seg_num, 
        #                        target_window=configs.pred_len, 
        #                        head_dropout=configs.fc_dropout)
        # print(final_seg_num)
         
        
        # # 为最后一层创建bottleneck
        # self.bottleneck = BottleneckLayer(
        #     d_model=configs.d_model,
        #     n_heads=configs.n_heads,
        #     d_ff=configs.d_ff,
        #     dropout=configs.dropout,
        #     seg_num=final_seg_num,
        #     configs=configs
        # )

    def forward(self, x,corr):
        # 进来的x 的维度   x: [B， C, pacth_number, dmodel] 这里是最开始分割的的patch
        # 存储encoder每一层层结果
        encode_x = []
        # 进来先存下来，第一个embedding
        x_head1=self.heads[0](x)
        # encode_x.append(x_head1)
        # print(x_head1.shape)
        # 进入 encoder层
        i=0
        for block in self.encode_blocks:
            x, attns = block(x,corr)

            # 展开
            x_head=self.heads[i](x)
            # 存结果 
            encode_x.append(x_head)
            # print(x_head.shape)
            i=i+1
        
        # 最后一层加上bottleneck
        # bottleneck_out = self.bottleneck(x, corr)
        # print('Bottle',bottleneck_out.shape)
        # encode_x.append(bottleneck_out)
        # print(encode_x[-2].shape) #torch.Size([128, 13, 1, 256])
        # print(encode_x[-1].shape) #torch.Size([128, 13, 1, 256])

        return encode_x, None


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # two stage TwoStageAttentionLayer
        self.self_attention = self_attention
        # 普通的 AttentionLayer
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
        # encoder结果先过一个 two stage 自注意力
        x = self.self_attention(x,corr)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        # 历史的信息做encoder x 做 Q  信息的做cross K V
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)


    def forward(self, x, cross_all,corr):

        ##############################added by yj######################################
        usebottle=False
        if usebottle==True:
            bottleneck_output = cross_all[-1]
            # 获取真正的encoder输出 (不包括原始输入和bottleneck)
            cross = cross_all[:-1]  
        else: 
            cross=cross_all
        ##############################added by yj######################################


        final_predict = None
        i = 0
        layers_pre=[]
        layernum=len(self.decode_layers)

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            if usebottle:
                if i==0:
                    cross_enc = cross[layernum-i-1]
                    # cross_enc = cross[i]
                    # print(bottleneck_output.shape)
                    # print(x.shape)
                    bottleneck_output=bottleneck_output+x[:,:,:bottleneck_output.size(2),:]
                    x, layer_predict = layer(x, bottleneck_output,corr)
                
                    # x_2, layer_predict_2= layer(x,bottleneck_output,corr)
                    # x=x_1+x_2
                    # layer_predict=layer_predict_1+layer_predict_2
                
                else:
                    cross_enc = cross[layernum-i-1]
                    x, layer_predict = layer(x, cross_enc,corr)
            else:
                cross_enc = cross[i]
                x, layer_predict = layer(x, cross_enc,corr)

            #------------变量保存------#
            predappend= rearrange(layer_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
            layers_pre.append(predappend)
            #------------变量保存------#  
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict,layers_pre



class BottleneckLayer(nn.Module):
    """
    Bottleneck layer for U-Net like architecture to aggregate information
    at the deepest level between encoder and decoder.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout,seg_num,configs):
        super(BottleneckLayer, self).__init__()

        self.last_enc=TwoStageAttentionLayer(configs, seg_num, configs.factor, d_model, n_heads, d_ff, dropout)
        
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

        # x_output=self.last_enc(x,corr)
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
    
    # TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
    #                                                          d_ff, dropout)



class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            # 这里的target_window就是predlength 直接展开到  pred length 去了，其实历史信息不需要这么多，我只添加一天的信息做输出就好
            # 记住下面是我的修改，我想要他嵌入embed，那么就是一个小维度的输出，进而尽可能不去影响未来信息 self.change_embed
            if self.change_embed:
                self.embed = nn.Sequential(nn.Linear(nf, self.change_embed_dim),
                                           nn.ReLU(),
                )
            else:
                self.linear = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                # print(z.shape)
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x