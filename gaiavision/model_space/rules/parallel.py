# standard lib
from collections.abc import Sequence
import warnings

# 3rd party lib
import pandas as pd

# local lib
from .base_rule import BaseRule, SAMPLE_RULES


@SAMPLE_RULES.register_module('parallel')
class Parallel(BaseRule):
    def __init__(self, rules):
        if not isinstance(rules, Sequence):
            raise TypeError(f'`rules` should be Sequence, but got `{type(rules)}`')
        elif len(rules) < 2:
            warnings.warn(f'Length of `rules` of `Parallel` is recommended to be greater than 1')
        self._rules = rules

    def _apply(self, obj):
        return [r(obj) for r in self._rules]


@SAMPLE_RULES.register_module('sliced_parallel')
class SlicedParallel(BaseRule):
    def __init__(self, rules):
        if not isinstance(rules, Sequence):
            raise TypeError(f'`rules` should be Sequence, but got `{type(rules)}`')
        elif len(rules) < 2:
            raise ValueError(f'length of `rules` of `Parallel` should be greater than 1')
        self._rules = rules

    def _apply(self, obj, index=None):
        return self._rules[index](obj)

    def __call__(self, obj, **kwargs):
        if isinstance(obj, Sequence):
            result = []
            assert len(obj) == len(self._rules)
            for i, v in enumerate(obj):
                if not isinstance(v, pd.DataFrame):
                    raise TypeError('`v` should be pandas.DataFrame, but got `{type(v)}`')
                else:
                    out = self._apply(v, index=i)
                if isinstance(out, Sequence):
                    result.extend(out)
                else:
                    result.append(out)
        else:
            raise TypeError('`obj` should be Sequence, but got `{type(obj)}`')
        return result


