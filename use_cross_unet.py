# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling use_cross_unet or otherwise documented as
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


import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# 可执行的main函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str,  default='Cross_Unet',
                        help='model name, options: [PatchTST, iTransformer, PaiFilter, CycleNet,PatchMLP,TimeMixer,TimesNet,Crossformer,Transformer,patchdecoder]')
    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='15T',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scaling', default=False, action="store_true", help='normalizing input data')
    parser.add_argument('--groupid',type=str,default='log train or test number',) # group num
    parser.add_argument('--station_name', type=str,  default='KDASC',help='Input station name')   # station name
    parser.add_argument('--deployment', default=False, action="store_true",help='True: Corrdiff for actual deployment; Flase: ordinary prediction')   # station name

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # iTransformer
    parser.add_argument('--output_attention_2', action='store_true', help='whether to output attention in ecoder in the added itransformer')
    parser.add_argument('--d_model_2', type=int, default=256, help='dimension of the added itransformer model')
    parser.add_argument('--dropout_2', type=float, default=0.05, help='dropout of the added itransformer model')
    parser.add_argument('--n_heads_2', type=int, default=8, help='num of heads of the added itransformer model')
    parser.add_argument('--d_ff_2', type=int, default=1024, help='dimension of fcn of the added itransformer model')
    parser.add_argument('--activation_2', type=str, default='gelu', help='activation of the added itransformer model')
    parser.add_argument('--e_layers_2', type=int, default=2, help='num of encoder layers of the added itransformer model')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # paifilter
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of the model')

    # CycleNet.
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')


    # Time mixer
    parser.add_argument('--channel_independence', type=int, default=1,help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--use_future_temporal_feature', type=int, default=1,
                        help='whether to use future_temporal_feature; True 1 False 0')

    # Tiems net
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    # PatchMLP
    parser.add_argument('--use_norm', default=False, action="store_true", help='use norm')
    parser.add_argument('--patch_len_arryay', nargs='+',type=int, default=[16], help='patch length list')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    
    # other basic args
    parser.add_argument('--weather_features_num', type=int, default=1, help='whether to use weather features') # weather number
    parser.add_argument('--history_num', type=int, default=1, help='whether to use weather features') # hist number
    parser.add_argument('--useweather', type=bool, default=True, help='use weather information') # use wather info
    # parser.add_argument('--use_satell', type=bool, default=True, help='use_satell information')  # 用于数据集添加辐射，  反之添加NWP预报
    parser.add_argument('--use_satell', action='store_true', help='use satellite data')
    # cross-Unet
    parser.add_argument('--usenonlinearproject', type=bool, default=False, help='nonlinear projection')
    parser.add_argument('--usebottle', type=bool, default=True, help='use bottleneck')
    parser.add_argument('--convmerge', type=bool, default=False, help='use convmerge')
    parser.add_argument('--swichchannel', type=bool, default=False, help=' swichchannel')
    parser.add_argument('--twofilter', type=bool, default=True, help=' twofilter for encoder and decoder')

    
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    ##########################################################################
    #  nohup python use_cross_unet.py > cross_solar_testing.log 2>&1 &

    args.is_training=1 # training

    for args.pred_len in [16,48,96,96*4,96*7]:# [16,48,96,96*4,96*7]
        # random seed--cycling
        fix_seed = args.random_seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # args.model= 'Cross_Unet' #select model
        # args.useweather=True # use wather forward helping

        if args.deployment:
            args.data='deployment'
            print('Corrdiff dataset for training actural predictions')
        else:
            args.data='custom'
            print('ordinary dataset for training')

        # channel design (variables)
        if args.station_name=='KDASC' or args.station_name=='yulara':
            args.weather_features_num=1 
            args.history_num=1             
        else:
            if args.use_satell==True or args.deployment==True:
                args.weather_features_num=1  # SWR
                args.history_num=7
            else:
                args.weather_features_num=6 
                args.history_num=7     
        
        # channel design
        if args.useweather:
            args.enc_in=args.history_num+args.weather_features_num # variabel input 
            args.dec_in=args.history_num+args.weather_features_num
            args.c_out=args.history_num+args.weather_features_num
        else:
            args.enc_in=args.history_num # variabel input 
            args.dec_in=args.history_num
            args.c_out=args.history_num

        # forecasting window
        if args.pred_len>95:
            args.seq_len=args.pred_len # gru model
        else:
            args.seq_len=96
        args.label_len=args.pred_len


        # other hyper parameteter
        args.e_layers=3 # encoder layer 
        args.factor=10
        args.d_ff=512
        args.d_model=256
        args.learning_rate=0.0001
        args.usenonlinearproject=True # use nonlinear projection  P-corr-module
        args.usebottle=True
        # seg length
        if args.pred_len<17:
            args.seg_len=12
        elif args.pred_len<49:
            args.seg_len=12
        elif args.pred_len<96*4+1:
            args.seg_len=24
        else:
            args.seg_len=48

        args.groupid=args.station_name
        args.task_name='long_term_forecast'  
        args.model_id='Formal-version-'+args.model+'-station-'+args.station_name 


        # fixed parameters
        # args.data='custom'
        args.features= 'S' #'MS'
        args.scaling=True
        args.inverse=True
        args.des='Exp'   
        args.itr=1
        args.patience=10
        args.train_epochs=100
        
        # other design parameters
        args.d_layers=args.e_layers+1
        args.n_heads=4
        args.fc_dropout=0.0
        args.head_dropout=0
        args.stride=4

        # checkpoint path
        args.checkpoints='./checkpoints/'+args.model+'/'+str(args.pred_len)+'/'

        Exp = Exp_Main
        print('Args in experiment:')
        print(args)

        # training
        if args.is_training:
            # running
            for ii in range(args.itr):
                # setting record of experiments
                setting = 'model_id{}_seq_len{}_stride{}_patch_len{}_Group{}'.format(
                    args.model_id,
                    args.seq_len,
                    args.stride,
                    args.patch_len,
                    args.groupid
                    )

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
        
        else:
            pass
        # test

        ii = 0
        setting = 'model_id{}_seq_len{}_stride{}_patch_len{}_Group{}'.format(
                args.model_id,
                args.seq_len,
                args.stride,
                args.patch_len,
                args.groupid)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
