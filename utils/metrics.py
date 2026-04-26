# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling metrics or otherwise documented as
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




import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    r2=calculate_r2_sklearn(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr,r2


def calculate_r2_sklearn(y_pred,y_true):
    # numpy flatten
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # sklearn
    r2 = r2_score(y_true, y_pred)
    
    return r2

def R2_yj(pred, true, scale = False, downsample = False):

    pred_ = pred
    true_ = true
    
    pred_ = pred_.flatten()
    true_ = true_.flatten()
    
    true_mean = np.mean(true_)
    SS_res = np.sum(np.power((true_ - pred_), 2))
    SS_tot = np.sum(np.power((true_ - true_mean), 2))
    
    r2 = 1 - SS_res / SS_tot
    return r2

def RSQ(pred, true, scale = False, downsample = False):
    """The RSQ function in Excel"""
    if downsample:
        pred_ = pred[::3]
        true_ = true[::3]
    else:
        pred_ = pred
        true_ = true
    pred_ = pred_.flatten()
    true_ = true_.flatten()
    true_mean = np.mean(true_)
    pred_mean = np.mean(pred_)
    
    r = np.dot(pred_ - pred_mean, true_ - true_mean) / np.sqrt(np.sum(np.power((pred_ - pred_mean),2)) * np.sum(np.power((true_ - true_mean),2)))
    return np.power(r, 2)




# def R2(pred, true, scale = False, downsample = False):
#     """reference: https://www.cnblogs.com/leezx/p/9929678.html"""

#     pred_ = pred
#     true_ = true
#     pred_ = pred_.flatten()
#     true_ = true_.flatten()
#     true_mean = np.mean(true_)
#     SS_res = np.sum(np.power((true_ - pred_), 2))
#     SS_tot = np.sum(np.power((true_ - true_mean), 2))
#     SS_reg = np.sum(np.power((pred_ - true_mean), 2))


#     if (1-SS_res/SS_tot - SS_reg/SS_tot) < 0.0001: 
#         # return 1-SS_res/SS_tot
#         return 1-SS_reg/SS_tot
#         # return -1000
#     print(f"r2: {1-SS_res/SS_tot}, {SS_reg/SS_tot}")
#     return -100


    
# def acc_day_ahead_mean(station_name, pred, true, scale = False, downsample = False, cap = 22620):
#     """reference: https://guangfu.bjx.com.cn/news/20180118/875022-2.shtml"""    
#     # if downsample:
#     #     pred_ = pred[::3]
#     #     true_ = true[::3]
#     # else:
#     pred_ = pred
#     true_ = true
    
#     if scale:
#         # cap = np.max(true_)
#         # 电站的设计容量，给的单位是MW，预测值是Kw
#         cap=20000.0
#     else:
#         cap=20000.0
#     # capnpw=20000.0
#     # capnpw=10000.0   #徐州
#     # capnpw=5980.0 # 龙口

#     if station_name=='桃园鑫辉':
#         capnpw=20000.0 
#     elif station_name=='龙口协鑫':
#         capnpw=5980.0
#     elif station_name=='莆田鑫能':
#         capnpw=15000.0
#     elif station_name=='徐州鑫日':
#         capnpw=15000.0
#     elif station_name=='桃源鑫能':
#         capnpw=20000.0
#     elif station_name=='桃园鑫源':
#         capnpw=20000.0
#     elif station_name=='长沙鑫佳':
#         capnpw=21500.0
#     else:
#         print('没有这个站点名称，检查下指标是否计算正确')
 
#     abs_diff = np.abs(pred_ - true_)
#     square = np.power(abs_diff, 2)
    
#     part = np.sum(square) / pred_.size
#     print(f"correct: {1 - np.sqrt(np.mean((pred_ - true_) ** 2)) / capnpw}")
#     return 1 - np.sqrt(part) / capnpw
#     # return 1 - np.sqrt(np.mean((pred_ - true_) ** 2)) / cap


# def acc_day_ahead(station_name, pred, true, scale = False, downsample = False, cap = 22620):
#     """reference: https://mguangfu.bjx.com.cn/mnews/20200421/1064899.shtml"""   
#     # if downsample:
#     #     pred_ = pred[::3]
#     #     true_ = true[::3]
#     # else:
#     pred_ = pred
#     true_ = true

#     if scale:
#         # cap = np.max(true_)
#         # 电站的设计容量，给的单位是MW，预测值是Kw
#         cap=20000.0
#     else:
#         cap=20000.0

#     if station_name=='桃园鑫辉':
#         capnpw=20000.0 
#     elif station_name=='龙口协鑫':
#         capnpw=5980.0
#     elif station_name=='莆田鑫能':
#         capnpw=15000.0
#     elif station_name=='徐州鑫日':
#         capnpw=15000.0
#     elif station_name=='桃源鑫能':
#         capnpw=20000.0
#     elif station_name=='桃园鑫源':
#         capnpw=20000.0
#     elif station_name=='长沙鑫佳':
#         capnpw=21500.0
#     else:
#         print('没有这个站点名称，检查下指标是否计算正确')
#     # capnpw=20000.0  # 桃源鑫辉  浦城鑫浦
#     # capnpw=15000.0  #  莆田新能
#     # capnpw=10000.0   #徐州
#     # capnpw=5980.0 # 龙口

#     abs_diff = np.abs(pred_ - true_)
#     square = np.power(abs_diff, 2)
#     abs_diff_sum = np.sum(abs_diff)
#     #cap = np.max(true)
#     part = np.sum(square * abs_diff / abs_diff_sum)
#     return 1 - np.sqrt(part) / capnpw



# def new_metric(station_name, pred, true, scale = False):
#     acc = acc_day_ahead(station_name,pred, true, scale)
#     acc_downsample = acc_day_ahead(station_name,pred, true, scale, True)
#     acc_mean = acc_day_ahead_mean(station_name, pred, true, scale)
#     acc_mean_downsample = acc_day_ahead_mean(station_name,pred, true, scale, True)
#     r2 = R2(pred, true, scale)
#     r2_downsample=calculate_r2_sklearn(pred, true)
#     rsq = RSQ(pred, true, scale)
#     rsq_downsample = RSQ(pred, true, scale, True)
#     return acc, acc_downsample, acc_mean, acc_mean_downsample, r2, r2_downsample, rsq, rsq_downsample