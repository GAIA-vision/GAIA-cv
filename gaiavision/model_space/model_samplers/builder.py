# mm library
from mmcv.utils import build_from_cfg

# local library
from .base_model_sampler import MODEL_SAMPLERS, BaseModelSampler
from .functional_model_sampler import (ConcatModelSampler, CompositeModelSampler,
                                       AnchorModelSampler, RepeatModelSampler)
from .random_model_sampler import RangeModelSampler, CandidateModelSampler


def build_model_sampler(cfg, default_args=None):
    if cfg['type'] == 'concat':
        model_sampler = ConcatModelSampler(
            [build_model_sampler(c, default_args) for c in cfg['model_samplers']])
    elif cfg['type'] == 'repeat':
        model_sampler = RepeatModelSampler(
            build_model_sampler(cfg['model_sampler'], default_args),
            times=cfg.get('times', None))
    elif cfg['type'] == 'composite':
        model_sampler = CompositeModelSampler(
            [build_model_sampler(c, default_args) for c in cfg['model_samplers']])
    else:
        model_sampler = build_from_cfg(cfg, MODEL_SAMPLERS, default_args)

    return model_sampler

