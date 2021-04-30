# standard lib
from collections.abc import Sequence

# 3rd party lib
import pandas as pd

# local library
from .base_rule import BaseRule, SAMPLE_RULES


def can_evaluate(func_str):
    try:
        eval(func_str)
        return True
    except SyntaxError:
        return False


@SAMPLE_RULES.register_module('eval')
class EvalRule(BaseRule):
    def __init__(self, func_str):
        if not isinstance(func_str, str):
            raise TypeError(f'`func_str` should be str, but got `{type(func_str)}`')
        if not can_evaluate(func_str):
            raise SyntaxError(f'`{func_str}` could not be evaluated, please check the syntax')

        self._func_str = func_str
        self._func = eval(func_str)

    def _apply(self, obj):
        return obj[obj.apply(self._func, axis=1)]

    def __repr__(self):
        return f'EvalRule(\'{self.func_str}\')'


if __name__ == '__main__':
    meta = {
        'scale': [400, 500, 600],
        'depth': [4, 6, 20],
        'latency': [3.5, 7.2, 9.8]
            }
    a = pd.DataFrame(meta)
    func_str = 'lambda x: x[\'depth\'] == 6'
    r = EvalRule(func_str)
    b = r(a)
    print(b)
