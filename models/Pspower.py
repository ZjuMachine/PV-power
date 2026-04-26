import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
   输入未来7天的radiant（也就是输入的weather）信息，因为未来7天的radiant信息和发电量高度相关
   想看看能达到多高的预测发电量精度
   此外可以选择性的加入时间戳，看看时间戳的影响
   通过1dcnn提取过去7天（15分钟间隔）的radiant信息，通过输出 3*7个参数来建模时间序列，7天的高斯参数
   看看是否能够进行建模
'''


class Model(nn.Module):
    def __init__(self, use_embedding=True):
        super().__init__()
        # 假设输入特征有 month, day, weekday, hour, minute
        # 可以酌情使用embedding或者简单线性层
        input_dim = 5  # month, day, weekday, hour, minute简单拼接而成
        input_dim = 5 + 7*96
        hidden_dim = input_dim*2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 输出 A, sigma, C
        )
    def forward(self, month, day, weekday, hour, minute,weather):
        # 归一化处理，例如：
        # month: 1-12 -> month_norm = (month-1)/11
        month_norm = (month - 1) / 11.0
        day_norm = (day - 1) / 30.0   # 如果天数最长31天，可以用31代替
        weekday_norm = weekday / 6.0
        hour_f = hour + minute /60.0
        hour_norm = hour_f / 23
        minute_norm= minute / 45
        
        inp = torch.stack([month_norm, day_norm, weekday_norm, hour_norm, minute_norm], dim=-1) 


        feats = torch.cat([inp, weather], dim=-1)
        # inp: [Batch, Length, 5]
        
        # 输出 A, sigma, C
        params = self.mlp(feats)  # [Batch, Length, 3]
        A = params[:,:, 0]      # [Batch, Length]
        sigma = params[:,:, 1]  # [Batch, Length]
        C = params[:,:,2]      # [Batch, Length]

        # 计算g_base
        # center hour at 12
        hour_centered = hour_f - 12.0
        g_base = A * torch.exp(-0.5 * (hour_centered / sigma).pow(2)) + C
        
        return g_base


class OneDCNNGaussian(nn.Module):
    """
    使用 1D CNN 提取过去 7 天辐照度信息，并输出 3*7=21 个参数
    分别对应 7 天的 (A_i, sigma_i, C_i).
    可选地将未来 7 天的 radiant 预测或时间戳信息一起融合。
    """
    def __init__(self, 
                 past_len=7*96,     # 过去 7 天，每天 96 个 15min 时间点
                 future_len=7*96,   # 如果需要接收未来 7 天 radiant，可自行调整
                 use_future_radiant=True,
                 use_time_feature=True,
                 cnn_channels=16,
                 kernel_size=3,
                 mlp_hidden=64):
        super().__init__()
        
        # 1) 1D CNN 提取过去 7 天的 radiant 特征
        #    假设输入形状 [Batch, 1, past_len]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 最终将序列特征聚合到 1 个时间步
        
        # 2) 对未来 7 天 radiant 或时间戳，可选地用线性层提取
        #    若不使用，可直接置 use_future_radiant=False, use_time_feature=False
        input_dim_mlp = 0
        if use_future_radiant:
            # 假设未来 7 天每 15 分钟一个点 -> 7*96=672
            # 也可以只是每日一个点 -> 7
            # 这里仅示例写 future_len
            self.future_linear = nn.Linear(future_len, mlp_hidden)
            input_dim_mlp += mlp_hidden
        
        if use_time_feature:
            # 假设时间戳特征: month, day, weekday, hour, minute => 5维
            # 或者你可能还有更多特征
            # 这里示例简单直接 linear，实际可embedding或其他处理
            self.time_linear = nn.Linear(5, mlp_hidden)
            input_dim_mlp += mlp_hidden
        
        # 用于将 CNN + (可选)未来radiant + (可选)时间戳 特征合并
        # 最终输出 3*7=21 个参数 (A1,B1,C1, ..., A7,B7,C7)
        # 你也可以改成 7 行 3 列的形状
        self.final_linear = nn.Linear(cnn_channels + input_dim_mlp, 21)
        
        # 记录
        self.use_future_radiant = use_future_radiant
        self.use_time_feature = use_time_feature
        self.cnn_channels = cnn_channels

    def forward(self, past_radiant, future_radiant=None, time_feature=None):
        """
        past_radiant: [Batch, 1, past_len], 过去 7天x96 点的辐照度
        future_radiant: [Batch, future_len], 若 use_future_radiant=True，输入未来7天的radiant
        time_feature: [Batch, 5]，若 use_time_feature=True，输入月、日、weekday、hour、minute等
        返回:
            abc_params: [Batch, 7, 3] 每个样本 7 天的 (A, sigma, C)
        """
        
        # (A) CNN 提取过去 7 天特征
        # 输入: [B, 1, past_len]
        x = F.relu(self.conv1(past_radiant))   # [B, cnn_channels, past_len]
        x = F.relu(self.conv2(x))              # [B, cnn_channels, past_len]
        x = self.pool(x)                       # [B, cnn_channels, 1]
        x = x.squeeze(-1)                      # [B, cnn_channels]
        
        # (B) 如果需要融合未来 radiant
        feat_list = [x]  # 先放进 CNN 提取的向量
        if self.use_future_radiant and future_radiant is not None:
            # future_radiant: [B, future_len]
            # -> [B, mlp_hidden]
            fut = self.future_linear(future_radiant)
            fut = F.relu(fut)
            feat_list.append(fut)
        
        # (C) 如果需要融合时间戳特征
        if self.use_time_feature and time_feature is not None:
            # time_feature: [B, 5]
            tf = self.time_linear(time_feature)
            tf = F.relu(tf)
            feat_list.append(tf)
        
        # (D) 拼接所有特征
        combine_feat = torch.cat(feat_list, dim=-1)  # [B, cnn_channels + ?]

        # (E) 最终线性层 -> 21 参数
        out = self.final_linear(combine_feat)  # [B, 21]

        # 解释为 7 组 (A, sigma, C)
        # reshape -> [B, 7, 3]
        abc_params = out.view(-1, 7, 3)

        return abc_params


def reconstruct_gaussian(abc_params, T=96, mu=48):
    """
    将 [Batch, Days, 3] 的 (A, sigma, C)
    重构为 [Batch, Days, T] 的序列曲线。
    """
    B, Days, _ = abc_params.shape
    device = abc_params.device
    
    # t: [T] = [0,1,...,95] 
    t = torch.arange(T, device=device).float()
    # t_shifted: [1, T] 方便批量广播
    t_shifted = (t - mu).unsqueeze(0)  # [1, T]
    
    # 构建输出张量 [B, Days, T]
    y_pred_all = torch.zeros(B, Days, T, device=device)
    
    for i_day in range(Days):
        A = abc_params[:, i_day, 0].unsqueeze(-1)      # [B, 1]
        sigma = abc_params[:, i_day, 1].unsqueeze(-1)  # [B, 1]
        C = abc_params[:, i_day, 2].unsqueeze(-1)      # [B, 1]
        
        # clamp sigma 避免负值或过小
        sigma = torch.clamp_min(sigma, 1e-5)    
        
        # 计算 exp(...) => shape [B,T]
        # (t_shifted: [1,T] 广播成 [B,T]；A, sigma, C: [B,1] 广播成 [B,T])
        gauss_t = A * torch.exp(-0.5 * (t_shifted / sigma).pow(2)) + C
        
        y_pred_all[:, i_day, :] = gauss_t
    
    return y_pred_all  # [B, Days, T]