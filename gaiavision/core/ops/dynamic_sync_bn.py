# standard lib
import pdb

# 3rd party lib
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import Function
from torch.nn import SyncBatchNorm
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
import numpy as np

# mm library
from mmcv.runner import get_dist_info

# local lib
from ..mixins import DynamicMixin


class DynamicSyncBatchNorm(SyncBatchNorm, DynamicMixin):
    r"""Batch normalization over multiple processes with mutable input channels.

    """
    N = 0
    group_by_size = {}
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, group_size=None):
        process_group = self._get_group(group_size)
        super(DynamicSyncBatchNorm, self).__init__(num_features,
           eps, momentum, affine, track_running_stats, process_group)
        # Record the upper bound of num_features
        self.max_num_features = num_features

    @staticmethod
    def _get_group(group_size):
        if group_size is None:
            return None

        if group_size in DynamicSyncBatchNorm.group_by_size:
            return DynamicSyncBatchNorm.group_by_size[group_size]
        else:
            rank, world_size = get_dist_info()
            assert world_size % group_size == 0
            num_groups = world_size / group_size
            rank_list = np.split(np.arange(world_size), num_groups)
            rank_list = [list(map(int, x)) for x in rank_list]
            groups = [dist.new_group(ranks) for ranks in rank_list]
            return groups[rank // group_size]

    def deploy_forward(self, input):
        active_num_features = input.size(1)
        self.running_mean.data = self.running_mean[:active_num_features]
        self.running_var.data = self.running_var[:active_num_features]
        self.weight.data = self.weight[:active_num_features]
        self.bias.data = self.bias[:active_num_features]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            exponential_average_factor,
            self.eps,
        )

    def forward(self, input):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(input)

        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('DynamicSyncBatchNorm expected \
                 input tensor to be on GPU')
        self._check_input_dim(input)
        active_num_features = input.size(1)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # currently tracking running stats with dynamic input sizes
        # is meaningless.
        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        r"""
        This decides whether the mini-batch stats should be used for
        normalization rather than the buffers. Mini-batch stats
        are used in training mode, and in eval mode when buffers
        are None.
        """
        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to vanilla BN when synchronization is not necessary
        if not need_sync:
            if self.running_mean is not None and self.running_var is not None:
                return F.batch_norm(
                    input,
                    self.running_mean[:active_num_features],
                    self.running_var[:active_num_features],
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.training or not self.track_running_stats,
                    exponential_average_factor,
                    self.eps,
                )
            else:
                return F.batch_norm(
                    input,
                    None,
                    None,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.training or not self.track_running_stats,
                    exponential_average_factor,
                    self.eps,
                )
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('DynamicSyncBatchNorm is only supported \
                     torch.nn.parallel.DistributedDataParallel')

            if self.track_running_stats:
                return sync_batch_norm.apply(
                    input,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.running_mean[:active_num_features],
                    self.running_var[:active_num_features],
                    self.eps,
                    exponential_average_factor,
                    process_group,
                    world_size)
            else:
                return sync_batch_norm.apply(
                    input,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    None,
                    None,
                    self.eps,
                    exponential_average_factor,
                    process_group,
                    world_size)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = DynamicSyncBatchNorm(
                module.num_features,
                module.eps, module.momentum,
                module.affine,
                module.track_running_stats,
                process_group)

            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias

            if module.track_running_stats:
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var
                module_output.num_batches_tracked = module.num_batches_tracked

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))

        del module
        return module_output

