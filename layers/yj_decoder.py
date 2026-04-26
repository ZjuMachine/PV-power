# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling SegmentRestoration or otherwise documented as
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

import torch
import torch.nn as nn
from einops import rearrange, repeat

class SegmentRestoration(nn.Module):
    def __init__(self, d_model, patch_len):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
    
        self.linear_proj = nn.Linear(d_model, patch_len)

    def forward(self, x):

        batch_size, ts_d, seg_num, d_model = x.shape

        if self.d_model== self.patch_len:
            projected_segments = x
        else:
            projected_segments = self.linear_proj(x)
        restored_sequence = projected_segments.reshape(batch_size, ts_d, -1)

        return restored_sequence