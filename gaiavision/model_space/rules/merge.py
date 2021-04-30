# standard lib
from collections.abc import Sequence
import logging

# 3rd party lib
import pandas as pd

# local library
from .base_rule import BaseRule, SAMPLE_RULES


@SAMPLE_RULES.register_module('merge')
class Merge(BaseRule):
    def __init__(self, reset_index=True):
        self._reset_index = reset_index

    def _apply(self, obj):
        pass

    def __call__(self, obj):
        if isinstance(obj, Sequence):
            out = pd.concat(obj, axis=0)
        elif isinstance(obj, pd.DataFrame):
            # logger.warning('...')
            out = obj

        # remove duplicated model points
        out = out.drop_duplicates()

        # whether reset index, e.g. (87, 8, 9, 20, 13) to (0, 1, 2, 3, 4)
        if self._reset_index:
            return out.reset_index()
        else:
            return out



