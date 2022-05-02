import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING


#x and y are [bs1,bs2,...,bk] x n
#interleave along feat dim
#output is [bs1,bs2,...,bk] x 2n
def interleave(x, y):
    z = torch.stack([x, y], dim=-1) #bs x n x 2
    z = z.flatten(-2) #bs x 2*n
    return z

#built from mmdetection and CondDETR implementations
#code is very different but computes the same thing
class SineTransform(nn.Module):
    def __init__(self, dim=128, scale=2*math.pi):
        super().__init__()
        assert dim % 2 == 0, 'embedding dim must be even'
        logspace = torch.logspace(start=1, end=-4, steps=dim//2, base=scale)
        self.register_buffer('logspace', logspace) #auto move to gpu/cpu

    #offset is [bs1,bs2,...,bk] x n 
    def forward(self, offset):
        pos = offset.unsqueeze(-1) * self.logspace #bs x dim/2
        pos = interleave(pos.sin(), pos.cos())
        return pos.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, dim=256, mode='add', out_proj=True):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.out_proj = None
        if out_proj:
            self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        pos = self.encode(x)
        if self.out_proj is not None:
            pos = self.out_proj(pos)
        return pos
        # if self.mode == 'add':
            # return x + pos
        # elif self.mode == 'cat':
            # return torch.cat((x, pos), dim=-1)
        # else:
            # print('mode %s not supported' % self.mode)
            # assert 1==2


@POSITIONAL_ENCODING.register_module()
class LearnedEncoding1d(PositionalEncoding):
    def __init__(self, num_embeds=100, dim=256, mode='add', out_proj=True):
        super().__init__(mode=mode, out_proj=out_proj)
        self.weight = nn.Embedding(num_embeds, dim).weight
        self.dim = dim

    def encode(self, x, *args, **kwargs):
        B, L, D = x.shape
        #pos = self.weight.expand(B, -1, -1)
        pos = self.weight.unsqueeze(0)
        return pos


@POSITIONAL_ENCODING.register_module()
class SineEncoding2d(PositionalEncoding):
    def __init__(self, dim=256, mode='add', out_proj=True, scale=2 * math.pi):
        super().__init__(mode=mode, out_proj=out_proj)
        self.dim = dim
        self.scale = scale
        self.sine_transform = SineTransform(dim//2, scale=scale)

    def encode(self, x):
        B, H, W, C = x.shape
        mask = torch.ones(H, W)
        mask = mask.to(dtype=torch.int, device=x.device)
        # mask = 1 - mask  # logical_not
        y = mask.cumsum(0, dtype=torch.float32)
        x = mask.cumsum(1, dtype=torch.float32)
        pos_y = self.sine_transform(y / y.max())
        pos_x = self.sine_transform(x / x.max())
        pos = torch.cat([pos_y, pos_x], dim=-1)
        pos = pos.unsqueeze(0)
        pos = pos.expand(B, -1, -1, -1)
        return pos
