# standard library

# mm library
import numpy as np
import pandas as pd
from mmcv.utils import build_from_cfg

# local library
from .base_rule import SAMPLE_RULES, BaseRule
from .sequential import Sequential
from .parallel import Parallel, SlicedParallel
from .sample import Sample
from .merge import Merge


def build_sample_rule(cfg, default_args=None):
    mapping = {
        'sequential': Sequential,
        'parallel': Parallel,
        'sliced_parallel': SlicedParallel,
    }

    if cfg.get('type', None) in ('sequential', 'parallel', 'sliced_parallel'):
        rule = mapping[cfg['type']](
            [build_sample_rule(c, default_args) for c in cfg['rules']])
    else:
        # NOTE: This may cause latent dangers. Instead, I choose to infer automatically.
        # rule_type = cfg.setdefault('type', 'eval')
        if 'type' not in cfg.keys():
            if 'func_str' in cfg.keys():
                cfg['type'] = 'eval'
            else:
                raise Exception(f'Only `eval` rule supports default type value, but got {cfg}')
        rule = build_from_cfg(cfg, SAMPLE_RULES, default_args)

    return rule
