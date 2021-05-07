"""Mixin classes for dynamic modules.
"""
# standard lib
from copy import deepcopy

# 3rd-parth lib
import torch.nn as nn

class DynamicMixin():
    """ Mixin defining all the operations to manipulate architecture of a module.
    The bussiness of it includes:
    -- manipulate the architecture of module
    -- demonstrate the current arch state of a dynamic module
    Inspired by duck-typing, the behaviour of arch manipulation is determined during runtime.
    """
    # def init_state(self):
    #     self._state = EasyDict()
    #     state = self._prepare_state()
    #     self._check_sanity(state)
    #     for k,v in state.items():
    #         self._state[k] = v

    # @abstractmethod
    # def _prepare_state(self, **kwargs):
    #     pass

    # def _check_sanity(self, state):
    #     valid_space = [v.split('manipulate_', 1)[-1] for v in dir(self) \
    #                    if v.startswith('manipulate_')]
    #     assert len(valid_space) > 0
    #     for key in state:
    #         assert key in valid_space, f"Invalid state({key}) for class({type(self)})"

    # def state(self):
    #     return self._state

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

        # self._deploying = mode
        # self.eval() # the module should be in eval mode during deploying
        # for module in self.children():
        #     if isinstance(module, DynamicMixin):
        #         module.deploy(mode)
        # return self
