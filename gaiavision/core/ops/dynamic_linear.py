# standard lib
import math

# 3rd party lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


class DynamicLinear(nn.Linear, DynamicMixin):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool=True) -> None:
        super().__init__(in_channels, out_channels, bias)
        self.cout_state = out_channels

    def manipulate_cout(self, cout: dict) -> None:
        self.cout_state = out_channels

    def deploy_forward(self, x: Tensor) -> Tensor:
        cin = x.size(1)
        self.weight.data = self.weight[:self.cout_state, :cin]
        self.bias.data = self.bias[:self.cout_state]
        return F.linear(x, self.weight, self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if getattr(self, '_deploying', False):
            return self.deploy_forward(input)

        cin = input.size(1)
        weight = self.weight[:self.cout_state, :cin]
        bias = self.bias[:self.cout_state]
        return F.linear(input, weight, bias)

