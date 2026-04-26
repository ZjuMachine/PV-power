# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling Exp_Main or otherwise documented as
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




from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST, iTransformeronly,PaiFilter,CycleNet,PatchMLP,TimeMixer,TimesNet,Crossformer,Transformer,Patchdecoder,Cross_Unet,TimeFilter
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop,visual_png
from utils.metrics import metric,R2_yj
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.masks = self._get_mask()
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformeronly,
            'PaiFilter':PaiFilter,
            'CycleNet': CycleNet,
            'PatchMLP':PatchMLP,
            'TimeMixer':TimeMixer,
            'TimesNet': TimesNet,
            'Crossformer': Crossformer,
            'Transformer': Transformer,
            'patchdecoder':Patchdecoder,
            'Cross_Unet':Cross_Unet,
            'TimeFilter': TimeFilter
        }
        model = model_dict[self.args.model].Model(self.args).float()
        print(self.args.model)
        print("111111111111111 the model:")
        print(model)         #summary(model,(576, 288, 36))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _get_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        vali_r2 =[]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_w, batch_w_mark, batch_cycle,time_mark,dec_w,seq_w_nwp_hist,seq_x_hist) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                batch_w = batch_w.float().to(self.device)
                batch_w_mark = batch_w_mark.float().to(self.device)
                dec_w=dec_w.float().to(self.device)
                seq_w_nwp_hist=seq_w_nwp_hist.float().to(self.device)
                seq_x_hist=seq_x_hist.float().to(self.device)
                if self.args.useweather==True:
                    dec_inp = torch.zeros(batch_y.size(0), self.args.pred_len, batch_y.size(-1)+dec_w.size(-1)).float().to(self.device)  
                    batch_y_label=batch_y[:, :self.args.label_len, :]
                    batch_y_combined=torch.cat([dec_w[:,-self.args.label_len:,:],batch_y_label],dim=-1).to(self.device)
                    dec_inp = torch.cat([batch_y_combined, dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)                    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Patch' in self.args.model:
                        outputs = self.model(batch_x, batch_w, batch_w_mark)
                    elif any(substr in self.args.model for substr in {'CycleNet'}):
                        outputs = self.model(batch_x, batch_cycle, batch_w)
                    elif  any(substr in self.args.model for substr in {'TimeMixer'}):
                        outputs = self.model(batch_x, batch_x_mark,  batch_y_mark,batch_w)
                    elif self.args.model == 'Transformer': 
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'patchdecoder'}):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'Cross_Unet'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist)          
                    elif any(substr in self.args.model for substr in {'Cross_head'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist)   
                    elif any(substr in self.args.model for substr in {'TimeFilter'}):
                        outputs,_  = self.model(batch_x, batch_w,self.masks, is_training=True)              
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,batch_w)
                f_dim = -1 if (self.args.features == 'MS') or (self.args.features == 'S') or (self.args.features == 'C') else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                r2_score=R2_yj(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
                vali_r2.append(r2_score)
        total_loss = np.average(total_loss)
        r2=np.average(vali_r2)
        self.model.train()
        return total_loss, r2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_r2=[]
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_w, batch_w_mark, batch_cycle,time_mark,dec_w,seq_w_nwp_hist,seq_x_hist) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                batch_w = batch_w.float().to(self.device)
                batch_w_mark = batch_w_mark.float().to(self.device)
                dec_w=dec_w.float().to(self.device)
                seq_w_nwp_hist=seq_w_nwp_hist.float().to(self.device)
                seq_x_hist=seq_x_hist.float().to(self.device)
                if self.args.useweather==True:
                    dec_inp = torch.zeros(batch_y.size(0), self.args.pred_len, batch_y.size(-1)+dec_w.size(-1)).float().to(self.device)  # 创建扩展后的零张量
                    batch_y_label=batch_y[:, :self.args.label_len, :]
                    batch_y_combined=torch.cat([dec_w[:,-self.args.label_len:,:],batch_y_label],dim=-1).to(self.device)
                    dec_inp = torch.cat([batch_y_combined, dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)                    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Patch' in self.args.model:
                            outputs = self.model(batch_x, batch_w, batch_w_mark)
                    elif any(substr in self.args.model for substr in {'CycleNet'}):
                        outputs = self.model(batch_x, batch_cycle, batch_w)
                    elif  any(substr in self.args.model for substr in {'TimeMixer'}):
                        outputs = self.model(batch_x, batch_x_mark,  batch_y_mark,batch_w)
                    elif self.args.model == 'Transformer': 
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'patchdecoder'}):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'Cross_Unet'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist) 
                    elif any(substr in self.args.model for substr in {'Cross_head'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist)   
                    elif any(substr in self.args.model for substr in {'TimeFilter'}):
                        outputs, moe_loss  = self.model(batch_x, batch_w,self.masks, is_training=True)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_w)
                    f_dim = -1 if (self.args.features == 'MS') or (self.args.features == 'S') or (self.args.features == 'C') else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model=='TimeFilter':
                        alpha =0.05
                        loss = criterion(outputs, batch_y)+ alpha * moe_loss
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    r2_score=R2_yj(outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
                    train_r2.append(r2_score)
                # print
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_r2=np.average(train_r2)
            vali_loss,val_r2 = self.vali(vali_data, vali_loader, criterion)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train R2: {3:.7f}| Val Loss: {4:.7f} Val R2: {5:.7f}".format(
                epoch + 1, train_steps, train_loss,train_r2,vali_loss,val_r2))
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        inputx = []
        time_marks=[]
        preds_reverscaling=[]
        trues_reverscaling=[]
        folder_path = './test_results/' +self.args.model+'/'+str(self.args.pred_len)+'/'+setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_w, batch_w_mark, batch_cycle,time_mark,dec_w,seq_w_nwp_hist,seq_x_hist) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                batch_w = batch_w.float().to(self.device)
                batch_w_mark = batch_w_mark.float().to(self.device)
                dec_w=dec_w.float().to(self.device)
                seq_w_nwp_hist=seq_w_nwp_hist.float().to(self.device)
                seq_x_hist=seq_x_hist.float().to(self.device)
                if self.args.useweather==True:
                    dec_inp = torch.zeros(batch_y.size(0), self.args.pred_len, batch_y.size(-1)+dec_w.size(-1)).float().to(self.device)  # 创建扩展后的零张量
                    batch_y_label=batch_y[:, :self.args.label_len, :]
                    batch_y_combined=torch.cat([dec_w[:,-self.args.label_len:,:],batch_y_label],dim=-1).to(self.device)
                    dec_inp = torch.cat([batch_y_combined, dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)                    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Patch' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Patch' in self.args.model:
                        outputs = self.model(batch_x, batch_w, batch_w_mark)
                    elif any(substr in self.args.model for substr in {'CycleNet'}):
                        outputs = self.model(batch_x, batch_cycle, batch_w)
                    elif  any(substr in self.args.model for substr in {'TimeMixer'}):
                        outputs = self.model(batch_x, batch_x_mark,  batch_y_mark,batch_w)
                    elif self.args.model == 'Transformer': 
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'patchdecoder'}):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_w)
                    elif any(substr in self.args.model for substr in {'Cross_Unet'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist) 
                    elif any(substr in self.args.model for substr in {'Cross_head'}):
                        outputs = self.model(batch_x, batch_x_mark,batch_w,dec_inp,seq_w_nwp_hist,seq_x_hist)   
                    elif any(substr in self.args.model for substr in {'TimeFilter'}):
                        outputs,_  = self.model(batch_x, batch_w,self.masks, is_training=True)   
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark,batch_w)
                f_dim = -1 if (self.args.features == 'MS') or (self.args.features == 'S') or (self.args.features == 'C') else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.scaling:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs_reversescaling = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_reversescaling = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                # non reversescaling
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # reversescaling
                outputs_reversescaling=outputs_reversescaling[:, :, f_dim:]
                batch_y_reversescaling=batch_y_reversescaling[:, :, f_dim:]
                # non reversescaling
                pred = outputs
                true = batch_y
                # reversescaling
                pred_reversescaling=outputs_reversescaling
                true_reversescaling=batch_y_reversescaling
                preds.append(pred)
                trues.append(true)
                preds_reverscaling.append(pred_reversescaling)
                trues_reverscaling.append(true_reversescaling)
                inputx.append(batch_x.detach().cpu().numpy())
                time_marks.append(time_mark.numpy())
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true_reversescaling[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred_reversescaling[0, :, -1]), axis=0)
                    visual_png(gt, pd, os.path.join(folder_path, str(i) + '.png'))
        # non reversescaling
        preds = np.concatenate(preds, axis=0) # (2688, 96, 1)
        trues = np.concatenate(trues, axis=0) # (2688, 96, 1)
        #  revsescaling
        preds_reverscaling=np.concatenate(preds_reverscaling, axis=0) # (2688, 96, 1)
        trues_reverscaling=np.concatenate(trues_reverscaling, axis=0) # (2688, 96, 1)
        inputx = np.array(inputx)
        time_marks=np.array(time_marks)
        print('time marks',time_marks.shape)
        print('inputx shape',inputx.shape)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        print('test rescaling shape:',preds_reverscaling.shape, trues_reverscaling.shape)
        preds_reverscaling=preds_reverscaling.reshape(-1, preds_reverscaling.shape[-2], preds_reverscaling.shape[-1])
        trues_reverscaling=trues_reverscaling.reshape(-1, trues_reverscaling.shape[-2], trues_reverscaling.shape[-1])
        print('testcrescaling shape:', preds_reverscaling.shape, trues_reverscaling.shape)
        # result save
        folder_path = './results/'+self.args.model+'/'+str(self.args.pred_len)+'/'+setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(f"preds shape: {preds.shape}, trues shape: {trues.shape}")
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        time_marks=time_marks.reshape(-1,time_marks.shape[-2],time_marks.shape[-1])
        trues = np.nan_to_num(trues, nan=0.0)
        preds=np.nan_to_num(preds, nan=0.0)
        preds_reverscaling=np.nan_to_num(preds_reverscaling, nan=0.0)
        trues_reverscaling=np.nan_to_num(trues_reverscaling, nan=0.0)
        mae, mse, rmse, mape, mspe, rse, corr ,r2= metric(preds, trues)
        mae_s,mse_s, rmse_s, mape_s, mspe_s, rse_s, corr_s ,r2_s=metric(preds_reverscaling, trues_reverscaling)
        print('-----------non reverse result-----------')
        print('R2: {},  MSE: {}, MAE: {}'.format(r2,mse, mae))
        print('-----------reverse result-----------')
        print('R2: {},  MSE: {}, MAE: {}'.format(r2_s,mse_s, mae_s))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('Non-reverse R2: {}, MSE: {}, MAE: {}\n'.format(r2, mse, mae))
        f.write('Reverse R2: {}, MSE: {}, MAE: {}\n'.format(r2_s, mse_s, mae_s))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'metrics.npy', np.array([r2,mse, mae]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'pred_s.npy', preds_reverscaling)
        np.save(folder_path + 'true_s.npy', trues_reverscaling)
        np.save(folder_path + 'x.npy', inputx)
        np.save(folder_path + 'time_mark.npy', time_marks)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_w) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return
