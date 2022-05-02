import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, ATTENTION,\
        FEEDFORWARD_NETWORK, POSITIONAL_ENCODING
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg

#@BLOCKS.register_module()
@FEEDFORWARD_NETWORK.register_module()
class DETRDecoder(BaseModule):
    def __init__(self,
            num_layers=6,
            self_attn_cfg=None,
            cross_attn_cfg=None,
            ffn_cfg=None,
            shared_norm_cfg=None,
            return_all_layers=False,
            init_cfg=None
        ):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.self_attns = [build_from_cfg(self_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.cross_attns = [build_from_cfg(cross_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.ffns = [build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK) for i in range(num_layers)]
        self.dim = self.self_attns[0].attn.qk_dim
        self.return_all_layers = return_all_layers
        
        self.self_attns = nn.ModuleList(self.self_attns)
        self.cross_attns = nn.ModuleList(self.cross_attns)
        self.ffns = nn.ModuleList(self.ffns)
        self.shared_norm = build_norm_layer(shared_norm_cfg, self.dim)[1]
    
    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, embeds_pos, feats, feats_pos, offset=0):
        output = []
        embeds = torch.zeros_like(embeds_pos)
        for i in range(self.num_layers):
            embeds = self.self_attns[i](embeds, embeds_pos)
            embeds = self.cross_attns[i](embeds, feats, embeds_pos, feats_pos, offset=offset)
            embeds = self.ffns[i](embeds)
            embeds = self.shared_norm(embeds)
            output.append(embeds.unsqueeze(0))
        output = torch.cat(output, dim=0)
        if self.return_all_layers:
            return output
        else:
            return output[-1]

        

@FEEDFORWARD_NETWORK.register_module()
class DETREncoder(BaseModule):
    def __init__(self,
            num_layers=6,
            self_attn_cfg=None,
            ffn_cfg=None,
            shared_norm_cfg=None,
            init_cfg=None
        ):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.self_attns = [build_from_cfg(self_attn_cfg, ATTENTION) for i in range(num_layers)]
        self.ffns = [build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK) for i in range(num_layers)]
        
        self.self_attns = nn.ModuleList(self.self_attns)
        self.ffns = nn.ModuleList(self.ffns)
    
    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, feats, feats_pos, offset=0):
        for i in range(self.num_layers):
            feats = self.self_attns[i](feats, feats_pos, offset=offset)
            feats = self.ffns[i](feats)
        return feats


