# standard lib
import math

# 3rd party lib
import torch
import torch.nn as nn
import torch.nn.functional as F

# mm lib
from mmcv.cnn import CONV_LAYERS

# local lib
from ..mixins import DynamicMixin


def make_divisible(v, divisor=8, min_value=None):
    """
    make `v` divisible
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@CONV_LAYERS.register_module('DynConv2d')
class DynamicConv2d(nn.Conv2d, DynamicMixin):
    CHANNEL_TRANSFORM_MODE = None # None or 1
    search_space = {'width'}

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 ):
        '''
        depthwise conv and regular conv work normally. Group conv not support yet.
        '''
        super(DynamicConv2d, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias)

        self.groups = groups
        self.max_in_channels = in_channels
        self.max_out_channels = out_channels
        self.width_state = out_channels

    def manipulate_width(self, width):
        assert isinstance(width, int), f'`active_channels` must be int now'
        assert width <= self.weight.size(0), f'`active_channels` exceed dim 0 of `weight`,' \
            f' {width} vs. {self.weight.size(0)}'
        self.width_state = width

    def deploy_forward(self, x):
        active_in_channels = x.size(1)
        self.weight.data = self.weight[:self.width_state, :active_in_channels, :, :]
        groups = active_in_channels if self.groups > 1 else 1
        if self.bias is not None:
            self.bias.data = self.bias[:self.width_state]
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, groups)

    def forward(self, x):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(x)

        active_in_channels = x.size(1)
        groups = active_in_channels if self.groups > 1 else 1
        weight = self.weight[:self.width_state, :active_in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.width_state]
        else:
            bias = None

        if x.dtype == torch.float32 and weight.dtype == torch.float16:
            x = x.half()

        y = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, groups)
        return y

    # TODO: replace the vanilla conv repr, instead of extra_repr
    def extra_repr(self):
        s = super(DynamicConv2d, self).extra_repr()
        s += f', max_channels={(self.max_in_channels, self.max_out_channels)}'
        s += f', width_state={self.width_state}'
        return s

