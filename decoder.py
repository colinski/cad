import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS, ATTENTION,\
        FEEDFORWARD_NETWORK, POSITIONAL_ENCODING
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
import cad
# from ..builder import BLOCKS

#@BLOCKS.register_module()
class Decoder(BaseModule):
    def __init__(self,
            num_layers=6,
            self_attn_cfg=None,
            cross_attn_cfg=None,
            feat_pos_cfg=None,
            embed_pos_cfg=None,
            ffn_cfg=None,
            init_cfg=None
        ):
        super(Decoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.self_attn = build_from_cfg(self_attn_cfg, ATTENTION)
        self.cross_attn = build_from_cfg(cross_attn_cfg, ATTENTION)
        self.ffn = build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK)

        self.feat_pos_encoding = build_from_cfg(feat_pos_cfg, POSITIONAL_ENCODING)
        self.embed_pos_encoding = build_from_cfg(embed_pos_cfg, POSITIONAL_ENCODING)
    
    def forward(self, embeds, feats):
        encoded_feats = self.feat_pos_encoding(feats)
        encoded_embeds = self.embed_pos_encoding(feats)
        import ipdb; ipdb.set_trace() # noqa
        embeds = self.self_attn(embeds, embeds, embeds)
        embeds = self.cross_attn(embeds, encoded_feats, feats)
        embeds = self.ffn(embeds)
        return embeds

if __name__ == '__main__':
    decoder = Decoder(
        self_attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8),
        cross_attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8),
        ffn_cfg=dict(type='SLP', in_channels=256),
        feat_pos_cfg=dict(type='SineEncoding2d', dim=256)
        embed_pos_cfg=dict(type='LearnedEncoding1d', num_embeds=100, dim=256)
    )
    feats = torch.randn(2, 28, 28, 256)
    embeds = torch.randn(2, 100, 256)
    out = decoder(embeds, feats)
    import ipdb; ipdb.set_trace() # noqa

