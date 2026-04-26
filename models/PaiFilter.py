import torch
import torch.nn as nn
from layers.RevIN_filternet import RevIN

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # 构造传进来
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        # 激活revin层
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        self.useweather=configs.useweather


    def circular_convolution(self, x, w):
        # 傅里叶变换 torch.fft.rfft返回正频率部分
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x_enc, x_mark_enc,w_enc):
        # print(x_enc.shape)

        if self.useweather:
            x=torch.cat([w_enc,x_enc],dim=-1)
        else:
            x=x_enc
        
        # print(x.shape)


        # print(x.shape)
        # torch.Size([128, 96, 13])
        # 不做时间戳信息
        z = x
        # 实例归一化
        z = self.revin_layer(z, 'norm')
        x = z
        # [B,L,N]
        x = x.permute(0, 2, 1)
        # [B,N,L]  傅里叶-->乘积->反向傅里叶
        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D
        # fc映射
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        # 反实例归一化
        z = self.revin_layer(z, 'denorm')
        x = z

        return x
