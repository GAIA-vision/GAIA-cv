"""Mixin classes for dynamic modules.
"""
# standard lib
from copy import deepcopy

# 3rd-parth lib
import torch.nn as nn

class DynamicMixin():
    """ Mixin defining all the operations to manipulate architecture of a module.
    Its bussiness includes:
    -- manipulating the architecture of module
    -- demonstrating the current arch state of a dynamic module
    Inspired by duck-typing, the behaviour of arch manipulation is determined during runtime.
    """
    # def init_state(self):
    #     raise NotImplementedError

    # def state(self):
    #     raise NotImplementedError

    def manipulate_arch(self, arch_meta):
        manipulate_fmt = 'manipulate_{}'
        assert isinstance(arch_meta, dict), f'`arch_meta` should be a dict,' \
                                            f' got {type(arch_meta)}'
        for k,v in arch_meta.items():
            manipulator = getattr(self, manipulate_fmt.format(k), None)
            if manipulator is not None:
                manipulator(v)
            else:
                raise Exception(f'`{k}` is not a supported operand of manipulator'
                                f' for `{self.__class__.__name__}`')

    def deploy(self, mode: bool = True):
        """ Sets the module in deploying mode.

        In deploying mode, the unsed parameters and buffers are abandoned after `deploy_forward`, and
        a clean state_dict is retained.
        """
        def _deploy(module, mode):
            if isinstance(module, DynamicMixin):
                module._deploying = mode
                for m in module.children():
                    _deploy(m, mode)
            elif isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                for m in module.children():
                    _deploy(m, mode)

        _deploy(self, mode)
        return self

