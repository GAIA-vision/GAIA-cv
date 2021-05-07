# standard lib
from collections.abc import Sequence
import warnings
import json
import textwrap

# 3rd party lib
import pandas as pd

# local lib
from .base_rule import BaseRule, SAMPLE_RULES


@SAMPLE_RULES.register_module('parallel')
class Parallel(BaseRule):
    ''' Apply sub-rules on input.
    When input is a dataframe, it applies each sub-rules on the dataframe.
    When input is a sequence, it applies each sub-rules on each element of input.
    '''
    def __init__(self, rules):
        if not isinstance(rules, Sequence):
            raise TypeError(f'`rules` should be Sequence, but got `{type(rules)}`')
        elif len(rules) < 2:
            warnings.warn(f'Length of `rules` of `Parallel` is recommended to be greater than 1')
        self._rules = rules

    def _apply(self, obj):
        return [r(obj) for r in self._rules]

    def __repr__(self):
        return self.__class__.__name__ + '([\n\t' + \
            ',\n\t'.join([repr(v) for v in self._rules]) + '\n])'


@SAMPLE_RULES.register_module('sliced_parallel')
class SlicedParallel(BaseRule):
    ''' Apply sub-rules on element of corresponding index in input.
    The number of rules shoule be equal to number of elements of input.
    '''
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

    def __repr__(self):
        return self.__class__.__name__ + '([\n\t' + \
            ',\n\t'.join([repr(v) for v in self._rules]) + '\n])'

