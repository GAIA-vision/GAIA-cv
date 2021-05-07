# standard lib
import collections
import bisect
from itertools import product
from copy import deepcopy

# 3rd party lib
import numpy as np

# mm lib
from mmcv.utils import build_from_cfg

# local lib
from .base_model_sampler import MODEL_SAMPLERS, BaseModelSampler


@MODEL_SAMPLERS.register_module('concat')
class ConcatModelSampler(BaseModelSampler):
    """ Model sampler as a concatentation of multiple model samplers.

    Args:
        model_samplers (sequence): List of model samplers to be concatenated
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = e._sample_len()
            r.append(l+s)
            s += l
        return r, s

    def __init__(self, model_samplers, mode='sample'):
        super(ConcatModelSampler, self).__init__(mode)
        assert isinstance(model_samplers, collections.abc.Sequence)
        assert len(model_samplers) > 0
        self._model_samplers = model_samplers
        self._cumulative_sizes, self._sample_length = self.cumsum(self._model_samplers)
        self._inner_counter = 0

        self._traverse_length = sum([ms._traverse_len() for ms in self._model_samplers])

    def _sample_len(self):
        return self._sample_length

    def _traverse_len(self):
        return self._traverse_length

    def reset(self):
        self._inner_counter = 0
        for ms in self._model_samplers:
            ms.reset()

    def sample(self):
        result = self.sample_by_index(self._inner_counter)
        self._inner_counter = (self._inner_counter + 1) % self._sample_length
        return result

    def sample_by_index(self, idx):
        ms_idx = bisect.bisect_right(self._cumulative_sizes, self._inner_counter)
        arch_meta = self._model_samplers[ms_idx].sample()
        return arch_meta

    # TODO: use a decorator to post-process the results of `yield from`
    def traverse(self):
        for ms in self._model_samplers:
            yield from ms.traverse()


@MODEL_SAMPLERS.register_module('composite')
class CompositeModelSampler(BaseModelSampler):
    """ Composite model sampler.
        Merge arch configs from each suborinate model samplers.
        Args:
            model_samplers (list): model samplers whose results are combined together.
    """
    def __init__(self, model_samplers, mode='sample'):
        assert isinstance(model_samplers, collections.abc.Sequence)
        assert len(model_samplers) > 1
        super(CompositeModelSampler, self).__init__(mode)
        for ms in model_samplers:
            if isinstance(ms, dict):
                ms = build_from_cfg(ms, MODEL_SAMPLERS)
                self._model_samplers.append(ms)
            elif callable(ms):
                self._model_samplers.append(ms)

        self._sample_length = self._model_samplers[0]._sample_len()
        self._traverse_length = np.prod([ms._traverse_len() for ms in self._model_samplers])

    def _sample_len(self):
        return self._sample_length

    def _traverse_len(self):
        return self._traverse_length

    def sample(self):
        arch_meta = {}
        for ms in self._model_samplers:
            meta = ms.sample()
            arch_meta.update(meta)
        return arch_meta

    def traverse(self):
        gens = [ms.traverse() for ms in self._model_samplers]
        cand_metas = product(*gens)
        for comb_metas in cand_metas:
            arch_meta = {}
            for meta in comb_metas:
                arch_meta.update(meta)
            yield arch_meta


@MODEL_SAMPLERS.register_module('anchor')
class AnchorModelSampler(BaseModelSampler):
    def __init__(self, anchors, mode='sample'):
        super(AnchorModelSampler, self).__init__(mode)
        self._anchors = collections.OrderedDict()
        anchors_ = deepcopy(anchors)
        for i, ac in enumerate(anchors_):
            name = ac.pop('name', str(i))
            if name in self._anchors:
                raise Exception(f'duplicated anchor name `{name}`')
            self._anchors[name] = ac
        self._order = list(self._anchors.keys())
        self.reset()

    def _sample_len(self):
        return len(self._anchors)

    def _traverse_len(self):
        return len(self._anchors)

    def reset(self):
        self._inner_counter = 0

    def sample_by_name(self, name):
        return self._anchors[name]

    def sample_by_index(self, idx):
        return self._anchors[self._order[idx]]

    def sample(self):
        result = self.sample_by_index(self._inner_counter)
        self._inner_counter = (self._inner_counter + 1) % len(self)
        return result

    def traverse(self):
        for a in self._anchors.values():
            yield a


@MODEL_SAMPLERS.register_module('repeat')
class RepeatModelSampler(BaseModelSampler):
    def __init__(self, model_sampler, mode='sample', times=None):
        super(RepeatModelSampler, self).__init__(mode)
        self._model_sampler = model_sampler
        self._model_samplers.append(model_sampler)
        if times is None:
            self._times = 1
        else:
            self._times = times
        self._sample_length = self._times * self._model_sampler._sample_len()
        self._traverse_length = self._times * self._model_sampler._traverse_len()

    @property
    def times(self):
        return self._times

    def _sample_len(self):
        return self._sample_length

    def _traverse_len(self):
        return self._traverse_length

    def reset(self):
        self.model_sampler.reset()

    def sample(self):
        return self._model_sampler.sample()

    def traverse(self):
        for _ in range(self._times):
            yield from self._model_sampler.traverse()

