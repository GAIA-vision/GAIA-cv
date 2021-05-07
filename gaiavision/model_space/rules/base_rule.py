# standard lib
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

# mm lib
from mmcv.utils import Registry

# 3rd party lib
import pandas as pd


SAMPLE_RULES = Registry('model sampling rules')


@SAMPLE_RULES.register_module('base')
class BaseRule(metaclass=ABCMeta):
    """ All subclasses should implement the followgin APIs:
    - _apply: apply rule on pd.DataFrame
    """
    @abstractmethod
    def _apply(self, obj, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __call__(self, obj):
        if isinstance(obj, Sequence):
            result = []
            for i, v in enumerate(obj):
                if not isinstance(v, pd.DataFrame):
                    raise TypeError('`v` should be pandas.DataFrame, but got `{type(v)}`')
                else:
                    out = self._apply(v)
                if isinstance(out, Sequence):
                    result.extend(out)
                else:
                    result.append(out)
            return result
        else:
            assert isinstance(obj, pd.DataFrame)
            return self._apply(obj)
