# standard lib
from collections.abc import Sequence

# 3rd party lib
import pandas as pd

# local library
from .base_rule import BaseRule, SAMPLE_RULES


@SAMPLE_RULES.register_module('sample')
class Sample(BaseRule):
    valid_operation = (
        'top', 'last', 'random'
    )
    def __init__(self, operation, key=None, value=1, mode='number'):
        assert operation in Sample.valid_operation
        self._operation = operation
        self._value = value
        if operation in ('top', 'last'):
            assert key is not None
        self._key = key
        if mode == 'ratio':
            assert value > 0 and value < 1
        elif mode == 'number':
            assert isinstance(value, int)
        self._mode = mode

    def _apply(self, obj):
        num_df = obj.shape[0]
        # parse number of series to be sampled
        if self._mode == 'number':
            num = int(self._value)
        elif self._mode == 'ratio':
            total = obj.shape[0]
            num = int(total * self._value)
        num = min(num_df, num)
        if num == 0:
            return obj

        if self._operation == 'top':
            return obj.nlargest(num, self._key)
        elif self._operation == 'last':
            return obj.nsmallest(num, self._key)
        elif self._operation == 'random':
            if obj.shape[0] == 0:
                return obj
            return obj.sample(num, axis=0)


if __name__ == '__main__':
    import numpy as np
    from pprint import pprint
    from .eval_rule import EvalRule
    from .parallel import Parallel
    from .sequential import Sequential
    meta = {
        'scale': np.arange(0, 1000, 100),
        'depth': np.arange(2, 22, 2),
        'latency': np.linspace(0, 20, 10),
    }
    a = pd.DataFrame(meta)
    # rule1
    func_strs = [
        'lambda x: x[\'scale\'] >= 300',
        'lambda x: x[\'scale\'] >= 400',
        'lambda x: x[\'scale\'] >= 500',
    ]
    rule1 = [EvalRule(s) for s in func_strs]
    para1 = Parallel(rule1)
    # rule2
    func_strs = [
        'lambda x: x[\'depth\'] >= 18',
        'lambda x: x[\'depth\'] >= 15',
        'lambda x: x[\'latency\'] <= 17',
    ]
    rule2 = [EvalRule(s) for s in func_strs]
    para2 = Parallel(rule2)
    sample = Sample('top', key='latency', n=1)
    print('--------------------- stage 1 ------------------------')
    rule = Sequential([para1, para2, sample])
    c = rule(a)
    pprint(c)




