# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling data_provider or otherwise documented as
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



from data_provider.data_loader import  Dataset_Custom, Dataset_Pred,Dataset_Deployment
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'deployment':Dataset_Deployment
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    # timef
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size =args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        # all batch
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=args.scaling,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle,
        station_name=args.station_name,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
