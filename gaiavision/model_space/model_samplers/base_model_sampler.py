# standard lib
from abc import ABCMeta, abstractmethod
from collections import deque, namedtuple
from itertools import product
from copy import deepcopy
import random

# 3rd party lib
import torch
import torch.nn as nn
import numpy as np

# mm lib
from mmcv.utils import Registry

MODEL_SAMPLERS = Registry('model sampler')


@MODEL_SAMPLERS.register_module('base')
class BaseModelSampler(metaclass=ABCMeta):
    """ All subclasses should implement the following APIs:
    - sample: sample elements in a certain pattern
    - traverse: sample elements one by one
    """
    modes = {'sample', 'traverse'}

    def __init__(self, mode='sample'):
        self._mode = mode
        self._model_samplers = []

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @abstractmethod
    def traverse(self, **kwargs):
        pass

    # Length of sample action. It is exhausting to get length of traverse action.
    def __len__(self):
        if self._mode == 'sample':
            return self._sample_len()
        elif self._mode == 'traverse':
            return self._traverse_len()
        else:
            raise NotImplementedError

    @abstractmethod
    def _sample_len(self):
        pass

    @abstractmethod
    def _traverse_len(self):
        pass

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        assert mode in BaseModelSampler.modes
        for ms in self._model_samplers:
            ms.set_mode(mode)
        self._mode = mode

    def __call__(self, **kwargs):
        action = getattr(self, self._mode, None)
        if action:
            return action(**kwargs)
        else:
            raise NotImplementedError

