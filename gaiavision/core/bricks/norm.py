# 3rd parth library
import torch.nn as nn
import torch.distributed as dist

# mm library
from mmcv.utils import is_tuple_of
from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm, _InstanceNorm
from mmcv.cnn import NORM_LAYERS
from mmcv.cnn.bricks.norm import infer_abbr
from mmcv.runner import get_dist_info

# local library
from ..ops.dynamic_bn import DynamicBatchNorm
from ..ops.dynamic_sync_bn import DynamicSyncBatchNorm
from ..ops.dynamic_naive_sync_bn import DynamicNaiveSyncBatchNorm


NORM_LAYERS.register_module('DynBN', module=DynamicBatchNorm)
NORM_LAYERS.register_module('DynSyncBN', module=DynamicSyncBatchNorm)
NORM_LAYERS.register_module('DynNaiveSyncBN', module=DynamicNaiveSyncBatchNorm)


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' or layer_type == 'DynSyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

