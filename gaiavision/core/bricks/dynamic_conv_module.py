# standard lib
import warnings

# mm lib
from mmcv.cnn import ConvModule, PLUGIN_LAYERS

# local lib
from .norm import build_norm_layer
from ..mixins import DynamicMixin


@PLUGIN_LAYERS.register_module()
class DynamicConvModule(ConvModule, DynamicMixin):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(DynamicConvModule, self).__init__(in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                groups=groups,
                                                bias=bias,
                                                conv_cfg=conv_cfg,
                                                norm_cfg=norm_cfg,
                                                act_cfg=act_cfg,
                                                inplace=inplace,
                                                with_spectral_norm=with_spectral_norm,
                                                padding_mode=padding_mode,
                                                order=order)
        # init active out_channels
        self.width_state = out_channels
        # rebuild normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # Use msra init by default and init again
        self.init_weights()

    def manipulate_width(self, width):
        # assert issubclass(self.conv, DynamicMixin), '`self.conv` should inherit from DynamicMixin' 
        self.conv.manipulate_width(width)


