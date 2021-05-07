# standard lib
from functools import partial

# 3rd-party lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import Function

# local lib
from ..mixins import DynamicMixin

TORCH_VERSION = torch.__version__


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class DynamicNaiveSyncBatchNorm(nn.BatchNorm2d, DynamicMixin):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).
    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """
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

        # prepare dynamic features
        active_num_features = input.size(1)
        running_mean = self.running_mean[:active_num_features]
        running_var = self.running_var[:active_num_features]
        weight = self.weight[:active_num_features]
        bias = self.bias[:active_num_features]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if not self.training:
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

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        running_mean += self.momentum * (mean.detach() - running_mean)
        running_var += self.momentum * (var.detach() - running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = weight * invstd
        bias = bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
