# 3rd parth lib
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import Function
from torch.nn import BatchNorm2d
import numpy as np

# local lib
from ..mixins import DynamicMixin

class DynamicBatchNorm(BatchNorm2d, DynamicMixin):
    r"""Batch normalization with mutable input channels.

    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(DynamicBatchNorm, self).__init__(num_features,
           eps, momentum, affine, track_running_stats)
        # Record the upper bound of num_features
        self.max_num_features = num_features

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


