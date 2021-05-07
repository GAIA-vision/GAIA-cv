# standard lib
from collections.abc import Sequence
import json

# 3rd party lib
import pandas as pd

# local lib
from .base_rule import BaseRule, SAMPLE_RULES


@SAMPLE_RULES.register_module('sequential')
class Sequential(BaseRule):
    def __init__(self, rules):
        self._rules = rules

    def _apply(self, obj):
        for r in self._rules:
            obj = r(obj)
        return obj

    def __repr__(self):
        return self.__class__.__name__ + '([\n\t' + \
            ',\n\t'.join([repr(v) for v in self._rules]) + '\n])'


if __name__ == '__main__':
    import numpy as np
    from pprint import pprint
    from .eval_rule import EvalRule
    from .parallel import Parallel
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
    print('--------------------- stage 1 ------------------------')
    rule = Sequential([para1, para2])
    c = rule(a)
    pprint(rule)
    # pprint(c)

