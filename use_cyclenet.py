import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--deployment', default=False, action="store_true",help='True: Corrdiff for actual deployment; Flase: ordinary prediction')   # station name

    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str,  default='PatchTST',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--val_data_path', type=str, default='ETTh1.csv', help='val data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='15T',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scaling', default=False, action="store_true", help='normalizing input data')
    parser.add_argument('--weather_features', type=int, default=1, help='number of weather features')
    # down smapling to 15min
    parser.add_argument('--downsample', default=False, action="store_true", help='downsample the input data')
    parser.add_argument('--weather_diff',default=False, action="store_true", help='use diff of weather forecast')
    parser.add_argument('--add_history_data',default=False, action="store_true", help='add_history_data')
    parser.add_argument('--add_future_data',default=False, action="store_true", help='add_future_data')

    parser.add_argument('--test_data_path', type=str, default='None test data',
                        help='Input data set name')
    parser.add_argument('--groupid',type=str,default='log train or test number',)
    parser.add_argument('--station_name', type=str,  default='KDASC',help='Input station name')    
    parser.add_argument('--skip_day_test',default=False, action="store_true", help='Test data skip one day')
    parser.add_argument('--skip_day_train',default=False, action="store_true", help='Train data skip one day')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--addmasknight',default=False, action="store_true", help='Add mask night')
    parser.add_argument('--divide_design_power',default=False, action="store_true", help='Use the design power of station to scale instead')
    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
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
    parser.add_argument('--add_cross_channel_attention',default=False, action="store_true", help='Use the channel convolution attention')
    parser.add_argument('--change_embed', default=False, action="store_true", help='Change Patchtst embed dim')
    parser.add_argument('--use_head_embed', default=False, action="store_true", help='use_head_embed')
    parser.add_argument('--change_embed_dim', type=int, default=96, help='mlp_embed_layer')
    parser.add_argument('--mlp_embed_layer', type=int, default=2, help='mlp_embed_layer')
    parser.add_argument('--n_head_embed', type=int, default=4, help='n_head_embed')
    parser.add_argument('--dmodel_embed', type=int, default=64, help='d__embed_model')
    # iTransformer
    parser.add_argument('--add_itransformer', action='store_true', help='add itranformer after patchtst', default=False)
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
    parser.add_argument('--weather_features_num', type=int, default=1, help='whether to use weather features')
    parser.add_argument('--history_num', type=int, default=1, help='whether to use weather features')
    parser.add_argument('--useweather', type=bool, default=True, help='use weather information')
    parser.add_argument('--use_satell', action='store_true', help='use satellite data')
    args = parser.parse_args()
    # random seed-
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
    args.is_training=1 
    for args.pred_len in [16,48,96,96*4,96*7]:# [16,48,96,96*4,96*7]:   #[16,48,96,96*4,96*7]:
        # args.station_name='8'  # 0  1 2  4 7  8 
        # args.pred_len=96
        args.model= 'CycleNet' #,'Transformer'  # 'PatchTSTfusion'    #'PatchTST'  'other_paper'

        if args.deployment:
            args.data='deployment'
            print('Corrdiff dataset for training actural predictions')
        else:
            args.data='custom'
            print('ordinary dataset for training')

        args.encoder_type=None 
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
        if args.pred_len>95:
            args.seq_len=args.pred_len # gru model
        else:
            args.seq_len=96
        args.label_len=args.pred_len
        args.useweather=True
        if args.useweather:
            args.enc_in=args.history_num+args.weather_features_num  
            args.dec_in=args.history_num+args.weather_features_num
            args.c_out=args.history_num+args.weather_features_num
        else:
            args.enc_in=args.history_num 
            args.dec_in=args.history_num
            args.c_out=args.history_num

        args.d_model =256 
        if args.pred_len<97:
            args.cycle_len=args.pred_len//2 
        else:
            args.cycle_len=96
    ##########################################################################
        args.groupid=args.station_name
        args.task_name='long_term_forecast' 
        args.model_id='Formal6-'+args.model+'-station-'+args.station_name  #'Try_tcnlstm'   #'Try_tcn'   #'Try_bilstm'  #'Try_lstm'     #'Try_cnn_lstm'       # 'Try_1dcnn'   #'Try_1dcnn'  'Try_gru'
        args.learning_rate=0.0001
        # args.data='custom'
        args.features= 'S' #'MS'
        args.scaling=True
        args.inverse=True
        # args.scale=False
        args.des='Exp' 
        args.itr=1
        args.patience=10
        args.train_epochs=100

        args.e_layers=3
        args.d_layers=1
        args.factor=3  
        args.n_heads=4
        args.fc_dropout=0.0
        args.head_dropout=0
        args.stride=4
        args.add_itransformer=False
        args.addmasknight=False
        
        if args.model== 'Patchsolar':
            args.add_cross_channel_attention=True
            args.change_embed=True
            args.change_embed_dim=96
            args.use_head_embed=True
            args.n_head_embed=4
            args.dmodel_embed=46
        else: 
            pass
        args.checkpoints='./checkpoints/'+args.model+'/'+str(args.pred_len)+'/'
        Exp = Exp_Main
        print('Args in experiment:')
        print(args)
        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = 'model_id{}_seq_len{}_stride{}_patch_len{}_add_itransformer{}_add_history_data{}_Group{}'.format(
                    args.model_id,
                    args.seq_len,
                    args.stride,
                    args.patch_len,
                    args.add_itransformer,
                    args.add_history_data,
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
        ii = 0
        setting = 'model_id{}_seq_len{}_stride{}_patch_len{}_add_itransformer{}_add_history_data{}_Group{}'.format(
                args.model_id,
                args.seq_len,
                args.stride,
                args.patch_len,
                args.add_itransformer,
                args.add_history_data,
                args.groupid)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
