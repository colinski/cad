import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, FEEDFORWARD_NETWORK
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..util import Residual
# from ..builder import BLOCKS

#single layer perceptron
@FEEDFORWARD_NETWORK.register_module()
class SLP(BaseModule):
    def __init__(self,
                 in_channels=256,
                 expansion_ratio=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 dropout_cfg=dict(type='DropPath', drop_prob=0.0),
                 init_cfg=None
        ):
        super(SLP, self).__init__(init_cfg)
        hidden_dim = int(in_channels * expansion_ratio)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            build_from_cfg(act_cfg, ACTIVATION_LAYERS),
            nn.Linear(hidden_dim, in_channels)
        )
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
    
    # def init_weights(self):
        # pass
    
    # def _forward(self, x):
        # return self.layers(x)

    def forward(self, x):
        identity = x
        x = self.layers(x)
        x = self.dropout(x)
        x = x + identity
        x = self.norm(x)
        return x
