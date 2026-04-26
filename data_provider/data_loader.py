# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling data_loader or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os, re, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import joblib
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import json

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args,root_path, flag='train', size=None,
                 features='S', 
                 target='active_power', scale=True, timeenc=0, freq='h',cycle=None,station_name=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.station_name=station_name

        self.root_path = root_path
        self.cycle = cycle
        self.use_satell=args.use_satell
        self.__read_data__()

    def __read_data__(self):
        # scaler #
        self.scaler = StandardScaler()
        self.scaler_inverse= StandardScaler()
        self.scaler2 = StandardScaler() #weather norm

        # load data
        file_directory=self.root_path+self.station_name+'.csv'
        df_raw= pd.read_csv(file_directory)

        # predicted PV power
        if self.station_name=='KDASC' or self.station_name=='yulara':
            self.target='Active_Power'
        else:
            self.target='power' 
        
        time_colu_name='Time' # the name of time column 

        # station variables
        if self.station_name=='KDASC' or self.station_name=='yulara':
            radiant_cols=['SWR'] 
            his_cols=[self.target]
        else:
            if self.use_satell:
                radiant_cols=['SWR']  # weather column
                his_cols=['lmd_totalirrad','lmd_diffuseirrad','lmd_temperature','lmd_pressure','lmd_winddirection','lmd_windspeed',self.target]
            else:
                radiant_cols=['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity','nwp_windspeed','nwp_winddirection']
                his_cols=['lmd_totalirrad','lmd_diffuseirrad','lmd_temperature','lmd_pressure','lmd_winddirection','lmd_windspeed',self.target]          
        # last combined data
        raw_cols = [time_colu_name] + radiant_cols + his_cols
        df_raw=df_raw[raw_cols]

        # create datetime column
        df_raw[time_colu_name]=pd.to_datetime(df_raw[time_colu_name])

        # recreate column
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(time_colu_name)
        df_raw = df_raw[[time_colu_name] + cols + [self.target]]


        ###------------------divide data start------------------###
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        ###------------------divide data start------------------###


        ###------------------scaling start-----------------###
        df_data = df_raw[his_cols]
        if self.scale:
            # get train data
            train_data = df_data[border1s[0]:border2s[0]]
            # train data fit 
            self.scaler.fit(train_data.values)
            self.scaler_inverse.fit(train_data.iloc[:,-1:].values)
            # all data apply 
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        ###------------------scaling end-----------------###


        # ----------------------------------get time information start ---------------------------------- # 
        df_stamp = df_raw[[time_colu_name]][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp[time_colu_name])


        # restore real time without norm : for further visualization 
        df_stamp['datetime'] = pd.to_datetime(df_stamp['date'])
        df_stamp['year']    = df_stamp['date'].dt.year
        df_stamp['month']   = df_stamp['date'].dt.month
        df_stamp['day']     = df_stamp['date'].dt.day
        df_stamp['hour']    = df_stamp['date'].dt.hour
        df_stamp['minute']  = df_stamp['date'].dt.minute
        self.data_sample_realtime = df_stamp[['year','month','day','hour','minute']].values # len, 5

        # restore norm time matrix


        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        '''
        get return time embedding
        【
        【year month day hour min】
        【year month day hour min】
        】
        '''
        # ----------------------------------get time information start ---------------------------------- # 
        self.data_stamp = data_stamp

        #----------------------------------handle weather data ----------------------------------# 
        # read weather data
        weather_data=df_raw[radiant_cols]

        # scale
        if self.scale:
            # get traindata
            train_data_w =weather_data[border1s[0]:border2s[0]]
            # train data fit
            self.scaler2.fit(train_data_w.values)
            # norm all data
            w_data = self.scaler2.transform(weather_data.values)
        else:
            w_data = weather_data.values
         #----------------------------------handle weather data ----------------------------------# 

        # final get 
        self.w_data = w_data[border1:border2] # weather 
        self.data_x = data[border1:border2] # ready to be input
        self.data_y = data[border1:border2] # ready to be output


        # if cycle is 5, you can get like: 0 1 2 3 4 0 1 2  cycle index
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


    def __getitem__(self, index):

        # get index
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        w_end=s_end+self.seq_len # forward-looking
    
        # get data      lanel len=pred len<= seq len
        seq_x = self.data_x[s_begin:s_end]  # hist input: index ---> index+ seqlen
        seq_y = self.data_y[r_begin:r_end]  # input+output: index+seqlen-label_len---> index+ seqlen                          
        seq_w = self.w_data[s_end:w_end]  # forward-looking weather:  index+seqlen--->index+seqlen+seqlen
        seq_x_mark = self.data_stamp[s_begin:s_end]  # corresponding time mark
        seq_y_mark = self.data_stamp[r_begin:r_end] # corresponding time mark
        seq_w_mark = self.data_stamp[s_end:w_end] # corresponding time mark
        # backward weather---> calculate P-corr matrix
        seq_w_nwp_hist=self.w_data[s_begin:s_end] #  index ---> index+ seqlen
        if index<self.seq_len:
            seq_x_hist=self.data_x[s_begin:s_end]
        else:
            seq_x_hist=self.data_x[s_begin-self.seq_len:s_begin]
        seq_w_dec=self.w_data[s_end:w_end]  # seq: seqlen+seqlen


        cycle_index = torch.tensor(self.cycle_index[s_end])   # cycle index
        time_mark=self.data_sample_realtime[s_begin:r_end] # all real time---> index+seqlen+prelen

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_w, seq_w_mark, cycle_index,time_mark,seq_w_dec,seq_w_nwp_hist,seq_x_hist

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler_inverse.inverse_transform(data)



class Dataset_Deployment(Dataset):
    def __init__(self, args,root_path, flag='train', size=None,
                 features='S', 
                 target='active_power', scale=True, timeenc=0, freq='h',cycle=None,station_name=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        if self.seq_len < self.pred_len:
            raise ValueError(f"seq_len({self.seq_len}) must >= pred_len({self.pred_len})")

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.station_name=station_name

        self.root_path = root_path
        self.cycle = cycle
        self.use_satell=args.use_satell
        self.__read_data__()

    def __read_data__(self):
        # scaler #
        self.scaler = StandardScaler()
        self.scaler_inverse= StandardScaler()
        self.scaler2 = StandardScaler() #weather norm

        # load data

        # file_directory=self.root_path+self.station_name+'.csv'
        # "C:\记录\NC修改20251013\20251202第一次修改\AIweatherdata\station_data"
        # file_directory='/mntlvlin20tb/yujzhang/cross-unet/corrdiff_dataset/station_data/'+self.station_name+'.csv'
        file_directory=self.root_path+'AIweatherdata/station_data/'+self.station_name+'.csv'
        df_raw= pd.read_csv(file_directory)

        # nwp数据路径文件夹
        # self.nwp_dir=f'/mntlvlin20tb/yujzhang/cross-unet/corrdiff_dataset/data/{self.station_name}'
        self.nwp_dir=self.root_path+f'AIweatherdata/data/{self.station_name}'

        # predicted PV power
        if self.station_name=='KDASC' or self.station_name=='yulara':
            self.target='Active_Power'
        else:
            self.target='power' 
        
        time_colu_name='Timestamp' # the name of time column 

        # station variables
        if self.station_name=='KDASC' or self.station_name=='yulara':
            radiant_cols=['SWR'] 
            his_cols=[self.target]
        else:
            radiant_cols=['ssrd_corrdiff']  # weather column
            his_cols=['lmd_totalirrad','lmd_diffuseirrad','lmd_temperature','lmd_pressure','lmd_winddirection','lmd_windspeed',self.target]       
        # last combined data
        # raw_cols = [time_colu_name] + radiant_cols + his_cols
        raw_cols = [time_colu_name] + his_cols # 除了预报其他的都弄好

        df_raw=df_raw[raw_cols]

        # create datetime column
        df_raw[time_colu_name]=pd.to_datetime(df_raw[time_colu_name])

        # recreate column
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(time_colu_name)
        df_raw = df_raw[[time_colu_name] + cols + [self.target]]


        ###------------------divide data start------------------###
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        ###------------------divide data start------------------###


        ###------------------scaling history+power data start-----------------###
        df_data = df_raw[his_cols]
        if self.scale:
            # get train data
            train_data = df_data[border1s[0]:border2s[0]]
            # train data fit 
            self.scaler.fit(train_data.values)
            self.scaler_inverse.fit(train_data.iloc[:,-1:].values) # make power scaler
            # all data apply 
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        ###------------------scaling history+power data end-----------------###


        # ----------------------------------get time information start ---------------------------------- # 
        df_stamp = df_raw[[time_colu_name]][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp[time_colu_name])


        # restore real time without norm : for further visualization 
        df_stamp['datetime'] = pd.to_datetime(df_stamp['date'])
        df_stamp['year']    = df_stamp['date'].dt.year
        df_stamp['month']   = df_stamp['date'].dt.month
        df_stamp['day']     = df_stamp['date'].dt.day
        df_stamp['hour']    = df_stamp['date'].dt.hour
        df_stamp['minute']  = df_stamp['date'].dt.minute
        self.data_sample_realtime = df_stamp[['year','month','day','hour','minute']].values # len, 5

        # restore norm time matrix

        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        '''
        get return time embedding
        【
        【year month day hour min】
        【year month day hour min】
        】
        '''
        # ----------------------------------get time information start ---------------------------------- # 
        self.data_stamp = data_stamp


        # final get 
        self.time_index = df_raw[time_colu_name].values  # 全局时间轴
        self.data_x = data[border1:border2] # ready to be input
        self.data_y = data[border1:border2] # ready to be output


        #----------------------------------read weather data ----------------------------------# 
        # read all weather csv file
        nwp_frames = [] # ---> [nwp_frame1, nwp_frame2, ...]
        for fp in sorted(glob.glob(os.path.join(self.nwp_dir, '*.csv'))):
            issue_time = _parse_issue_date_from_filename(fp)  # 当天00:00
            df_n = pd.read_csv(fp)

            df_n = df_n[[time_colu_name] + radiant_cols].copy()
            df_n[time_colu_name] = pd.to_datetime(df_n[time_colu_name])
            df_n['issue_time'] = issue_time
            df_n = df_n.rename(columns={time_colu_name: 'valid_time'})
            nwp_frames.append(df_n)
        
        nwp_all = pd.concat(nwp_frames, ignore_index=True)
        


        # ========= 1. 先处理负值：全部置 0 =========
        for col in radiant_cols:
            # 小于 0 的值替换成 0
            nwp_all[col] = nwp_all[col].where(nwp_all[col] >= 0, 0.0)

        # ========= 2. 再去掉极端大值（按分位数裁剪） =========
        clip_q = 0.99

        for col in radiant_cols:
            upper = nwp_all[col].quantile(clip_q)
            nwp_all[col] = nwp_all[col].clip(upper=upper)
        
        nwp_all = nwp_all.set_index(['issue_time', 'valid_time']) # set index col
        #----------------------------------read weather data ----------------------------------# 


        #-------------------------scale weather data ----------------# 
        if self.scale:
            # day (.floor('D')) in train data
            train_dates = pd.to_datetime(df_raw[time_colu_name].iloc[border1s[0]:border2s[0]]).dt.floor('D').unique()
            nwp_train = nwp_all.loc[(train_dates, slice(None)), radiant_cols] # filter use the issue_time
            self.scaler2.fit(nwp_train.values) # fit scale machine
        self.nwp_all = nwp_all  # original nwp data
        #-------------------------scale weather data ----------------# 


        #----------------------------------calculate valid indices ----------------------------------# 
        self.valid_indices = []
        total = len(self.data_x) # length of all the train/test/valid dataset
        b1=border1
        b2=border2
        for i in range(0, total - self.seq_len - self.seq_len + 1):
            g_s_begin = b1 + i
            g_s_end = g_s_begin + self.seq_len
            g_w_end = g_s_end + self.seq_len


            if g_w_end > len(self.time_index): # less than total time in the dataset
                continue

            # 边界2：当前参考时刻 t 的 issue 必须覆盖到 w_end-1 的valid_time
            t_ref = pd.Timestamp(self.time_index[g_s_end - 1])
            issue_time = t_ref.floor('D')  # 当天00:00的这次预报
            last_needed_vt = pd.Timestamp(self.time_index[g_w_end - 1])
            # 该 issue 的最大小时刻 = issue_time + 8天（含头不含尾安全起见 >= 即可）
            if last_needed_vt <= issue_time + pd.Timedelta(days=8):
                self.valid_indices.append(i)
        #----------------------------------calculate valid indices ----------------------------------# 



        # #----------------------------------handle weather data ----------------------------------# 
        # # read weather data
        # weather_data=df_raw[radiant_cols]

        # # scale
        # if self.scale:
        #     # get traindata
        #     train_data_w =weather_data[border1s[0]:border2s[0]]
        #     # train data fit
        #     self.scaler2.fit(train_data_w.values)
        #     # norm all data
        #     w_data = self.scaler2.transform(weather_data.values)
        # else:
        #     w_data = weather_data.values
        # #----------------------------------handle weather data ----------------------------------# 

        # final get 
        # self.w_data = w_data[border1:border2] # weather 


        # 记录列配置，供 __getitem__ 使用
        self.time_col = time_colu_name
        self.his_cols = his_cols
        self.radiant_cols = radiant_cols
        self.b1 = b1
        self.b2 = b2


        # if cycle is 5, you can get like: 0 1 2 3 4 0 1 2  cycle index
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]




    def __getitem__(self, index):

        raw_index=index
        index = self.valid_indices[index]

        # get index
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        w_end=s_end+self.seq_len # forward-looking


        # 全局时间索引（相对于完整序列）
        g_s_begin = self.b1 + s_begin
        g_s_end = self.b1 + s_end
        g_r_begin = self.b1 + r_begin
        g_r_end = self.b1 + r_end
        g_w_end = self.b1 + w_end
    
        # get data      lanel len=pred len<= seq len
        seq_x = self.data_x[s_begin:s_end]  # hist input: index ---> index+ seqlen
        seq_y = self.data_y[r_begin:r_end]  # input+output: index+seqlen-label_len---> index+ seqlen                          
        # seq_w = self.w_data[s_end:w_end]  # forward-looking weather:  index+seqlen--->index+seqlen+seqlen
        seq_x_mark = self.data_stamp[s_begin:s_end]  # corresponding time mark
        seq_y_mark = self.data_stamp[r_begin:r_end] # corresponding time mark
        seq_w_mark = self.data_stamp[s_end:w_end] # corresponding time mark


        # 参考时刻与未来有效时刻列表
        t_ref = pd.Timestamp(self.time_index[g_s_end - 1])  # 输入的历史的最后一个时间点
        issue_time = t_ref.floor('D') # 输入的历史的最后一个时间点所在的天

        #
        t_ref_his=pd.Timestamp(self.time_index[g_s_begin]) 
        prev_issue_time = t_ref_his.floor('D') # D-1 # 如果历史nwp不够，需要去前一天预报找

        future_vts = pd.to_datetime(self.time_index[g_s_end:g_w_end])  # 未来窗口（长度=seq_len）
        hist_vts = pd.to_datetime(self.time_index[g_s_begin:g_s_end])  # 历史窗口，用于 nwp_hist

        # 从 nwp_all 取该 issue 的未来窗口
        # MultiIndex重建：[(issue_time, vt) for vt in future_vts]
        idx_future = pd.MultiIndex.from_product([[issue_time], future_vts],
                                                names=['issue_time', 'valid_time'])
        w_future = self.nwp_all.reindex(idx_future)[self.radiant_cols].values
        # 核查
        if  np.isnan(w_future).any():
            raise ValueError(f"NWP data contains NaN for issue_time={issue_time}, valid_times={future_vts}")
        if self.scale:
            w_future = self.scaler2.transform(w_future)


        # 从 nwp_all 取该 issue 的历史预报窗口方便计算相似度
        idx_hist_weather = pd.MultiIndex.from_product([[prev_issue_time], hist_vts],
                                                names=['issue_time', 'valid_time'])
        w_hist = self.nwp_all.reindex(idx_hist_weather)[self.radiant_cols].values      
        if  np.isnan(w_hist).any():  
            raise ValueError(f"history NWP data contains NaN for issue_time={prev_issue_time}, valid_times={hist_vts}")  
        if self.scale:
            w_hist = self.scaler2.transform(w_hist)




        # # ============ 🔍 Debug: 只打印前几个样本 =============
        # if raw_index < 10:
        #     print("\n========== DEBUG SAMPLE ==========")
        #     print(f"raw_index           = {raw_index}")
        #     print(f"s_begin(local)      = {s_begin}, s_end(local) = {s_end}")
        #     print(f"g_s_begin(global)   = {g_s_begin}, g_s_end(global) = {g_s_end}")
        #     print(f"issue_time(D)       = {issue_time}")
        #     print(f"prev_issue_time(hist)= {prev_issue_time}")

        #     # 历史功率时间+反归一化值
        #     print("\n[Hist time & power (前5个)]")
        #     hist_times = pd.to_datetime(self.time_index[g_s_begin:g_s_end])
        #     # 假设 his_cols 最后一列是 target
        #     # seq_x 是 scaled 的，反归一化看得更直观
        #     power_norm = seq_x[:, -1:]  # (seq_len,1)
        #     power_real = self.inverse_transform(power_norm)[:, 0]
        #     for k in range(min(5, len(hist_times))):
        #         print(f"{hist_times[k]} | P={power_real[k]:.2f}")

        #     # 未来 NWP，对应 future_vts
        #     print("\n[Future vts & NWP raw (前5个)]")
        #     for k in range(min(5, len(future_vts))):
        #         vt = future_vts[k]
        #         nwp_raw = self.nwp_all.loc[(issue_time, vt)][self.radiant_cols].values
        #         print(f"vt={vt} | issue={issue_time} | nwp_raw={nwp_raw}")

        #     # 历史 NWP，对应 hist_vts
        #     print("\n[Hist vts & NWP raw (前5个)]")
        #     for k in range(min(5, len(hist_vts))):
        #         vt = hist_vts[k]
        #         nwp_hist_raw = self.nwp_all.loc[(prev_issue_time, vt)][self.radiant_cols].values
        #         print(f"vt={vt} | hist_issue={prev_issue_time} | nwp_hist_raw={nwp_hist_raw}")

        #     print("==================================\n")
        # # ============ 🔍 Debug 结束 =============

        # 返回位次对齐
        seq_w = w_future                          # forward-looking NWP
        seq_w_dec = w_future                      # 你原本就返回了同一段作为 decoder NWP
        seq_w_nwp_hist = w_hist                   # 历史段的 NWP

        if index < self.seq_len:
            seq_x_hist = self.data_x[s_begin:s_end]
        else:
            seq_x_hist = self.data_x[s_begin-self.seq_len:s_begin]

        cycle_index = torch.tensor(self.cycle_index[s_end])   # cycle index
        time_mark=self.data_sample_realtime[s_begin:r_end] # all real time---> index+seqlen+prelen

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_w, seq_w_mark, cycle_index,time_mark,seq_w_dec,seq_w_nwp_hist,seq_x_hist

    def __len__(self):
        return len(self.valid_indices)
        # return len(self.data_x) - self.seq_len - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler_inverse.inverse_transform(data)
    
    


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='activate_power', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_directory=self.root_path+self.station_name+'.csv'
        df_raw = pd.read_csv(file_directory)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''


        '''
        # df_raw.columns: ['date', ...(other features), target feature]
        df_raw.columns: ['date', 'activate_power', 'total_cap', 'temperature', 'radiant', 'current_radiant', (potential weather variants...)]
        '''

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        # 重新组合成顺序
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, None

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



def _parse_issue_date_from_filename(path):
    """
    从文件名里抽取日期：支持 
        - 2016-03-01.csv
        - 20160301.csv
        - nwp_2016_03_01.csv
        - 2019-05-17-00-00-00.csv  ← 新增支持
    统一返回 pandas.Timestamp(YYYY-MM-DD 00:00:00)
    """
    name = os.path.splitext(os.path.basename(path))[0]  # 去掉 .csv
    # 匹配至少三组连续的数字（年、月、日），允许用 - 或 _ 分隔
    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', name)
    if not m:
        raise ValueError(f"无法从文件名解析日期: {name}")
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return pd.Timestamp(f"{y}-{mo}-{d} 00:00:00")
 



 
        # # 历史段的 NWP（用于相关性等，按需使用）---> calculate P-corr matrix
        # mask_late  = hist_vts >= issue_time    # 当天及之后（通常就是当天）
        # mask_early = ~mask_late                # 比 issue_time 早的（前半段）

        # # 2.1 使用 D 取后半段（hist_vts >= D）
        # if mask_late.any():
        #     idx_hist_late = pd.MultiIndex.from_product(
        #         [[issue_time], hist_vts[mask_late]],
        #         names=['issue_time', 'valid_time']
        #     )
        #     w_hist_late = self.nwp_all.reindex(idx_hist_late)[self.radiant_cols].values # 拿出当前D里面的预报

        #     w_hist[mask_late] = w_hist_late
        # # 2.2 使用 D-1 取前半段（hist_vts < D），如果 D-1 不存在，就用当前 NWP 近似
        # if mask_early.any():
        #     # 如果 nwp_all 里有 D-1 这期 没有的话说明是数据集起点
        #     if prev_issue_time in self.nwp_all.index.get_level_values(0):
        #         idx_hist_early = pd.MultiIndex.from_product(
        #             [[prev_issue_time], hist_vts[mask_early]],
        #             names=['issue_time', 'valid_time']
        #         )
        #         w_hist_early = self.nwp_all.reindex(idx_hist_early)[self.radiant_cols].values
        #     else:
        #         # === 关键：数据集起点附近，没有 D-1 这期 NWP ===
        #         # 你的原始想法：历史 NWP 近似等于“这里的 NWP”，
        #         # 我这里直接用当前 issue 的 w_future[0] 来近似整个前半段
        #         w_hist_early = np.repeat(w_future[0:1, :], mask_early.sum(), axis=0)

        #     w_hist[mask_early] = w_hist_early

