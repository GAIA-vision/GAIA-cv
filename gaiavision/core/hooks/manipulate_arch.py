# standard lib
import os.path as osp
import warnings
from math import inf
from collections.abc import Sequence

# 3rd-party lib
from torch.utils.data import DataLoader
import torch.distributed as dist

# mm lib
import mmcv
from mmcv.runner import HOOKS, Hook, get_dist_info

# local lib
from ..mixins import DynamicMixin
from ...model_space.utils import fold_dict
from ...utils.dist_helper import broadcast_object


@HOOKS.register_module()
class ManipulateArchHook(Hook):
    """Hook class for architecture manipulation.

    Args:
        model_sampler (T): model sampler that provides model meta.
    """

    def __init__(self, model_sampler):
        self.model_sampler = model_sampler

    def sample(self, sync=True):
        arch_meta = self.model_sampler.sample()
        _, world_size = get_dist_info()
        if sync and world_size > 1:
            # broadcast master arch to all rank if is distributed.
            arch_meta = broadcast_object(arch_meta)
        return arch_meta

    def manipulate_arch(self, runner, arch_meta):
        if isinstance(runner.model, DynamicMixin):
            runner.model.manipulate_arch(arch_meta)
        elif isinstance(runner.model.module, DynamicMixin):
            runner.model.module.manipulate_arch(arch_meta)
        else:
            raise Exception('Current model does not support arch manipulation.')

    def before_train_iter(self, runner):
        model_meta = self.sample(sync=True)
        model_meta = fold_dict(model_meta)
        self.manipulate_arch(runner, model_meta['arch'])

