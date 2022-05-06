# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import build_from_cfg
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, FEEDFORWARD_NETWORK, ATTENTION
from mmcv.runner.base_module import BaseModule

@ATTENTION.register_module()
class ResSelfAttn(BaseModule):
    def __init__(self,
                 attn_cfg=None,
                 norm_cfg=dict(type='LN'),
                 dropout_cfg=dict(type='DropPath', drop_prob=0.0),
                 init_cfg=None
        ):
        super().__init__(init_cfg)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, attn_cfg['qk_dim'])[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
    
    def forward(self, x, x_pos=None, offset=0):
        identity = x
        encoded_x = x if x_pos is None else x + x_pos
        x = self.attn(encoded_x, encoded_x, x, offset=offset)
        x = self.dropout(x)
        x = identity + x
        x = self.norm(x)
        return x

@ATTENTION.register_module()
class ResCrossAttn(BaseModule):
    def __init__(self,
                 attn_cfg=None,
                 norm_cfg=dict(type='LN'),
                 dropout_cfg=dict(type='DropPath', drop_prob=0.0),
                 init_cfg=None
        ):
        super().__init__(init_cfg)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, attn_cfg['qk_dim'])[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
    
    def forward(self, x, feats, x_pos=None, feats_pos=None, offset=0):
        identity = x
        encoded_x = x if x_pos is None else x + x_pos
        encoded_feats = feats if feats_pos is None else feats + feats_pos
        x = self.attn(encoded_x, encoded_feats, feats, offset=offset)
        x = self.dropout(x)
        x = identity + x
        x = self.norm(x)
        return x
