# standard lib
import copy
import random
import itertools
from collections.abc import Sequence

# 3rd-parth lib
import numpy as np

# local lib
from .base_model_sampler import MODEL_SAMPLERS, BaseModelSampler


@MODEL_SAMPLERS.register_module('range')
class RangeModelSampler(BaseModelSampler):
    """ Range model sampler.
    Args:
        start (list): start values of items.
        end (list): end values of items
        step (list): step of items.
        ascending (bool): the latter elements should always be larger if True

    """
    def __init__(self, key, start, end, step, ascending=False, depth_uniform=False, mode='sample'):
        super(RangeModelSampler, self).__init__(mode)
        
        if 'depth' not in key:
            assert depth_uniform is False, "depth_uniform can't be used here"
            
        one_dim = True
        # infer ndim
        if isinstance(start, Sequence):
            one_dim = False
            assert isinstance(end, Sequence)
            assert isinstance(step, Sequence)
            assert len(start) == len(end) == len(step), f'`start, end, step`' \
                f'should share the same length, got {len(start)}, {len(end)}, {len(step)}'
        self.ndim = 1 if one_dim else 2

        self.key = key
        self.start = start
        self.end = end
        self.step = step
        self.ascending = ascending
        self.depth_uniform = depth_uniform
        if self.depth_uniform:
            self.depth_cands = self.enumeration()
        if self.ndim == 2:
            if self.ascending:
                n = 0
                candidates = []
                for start, end, step in zip(self.start, self.end, self.step):
                    candidates.append(list(range(start, end+1, step)))
                candidates = itertools.product(*candidates)
                for c in candidates:
                    valid = True
                    prev = 0
                    for v in c:
                        if v < prev:
                            valid = False
                            break
                        else:
                            prev = v
                    if valid:
                        n += 1
            else:
                n = 1
                for start, end, step in zip(self.start, self.end, self.step):
                    n *= (end - start + 1) // step
            self._traverse_length = n
        else:
            self._traverse_length = (self.end - self.start + 1) // self.step

    def _sample_len(self):
        return 1

    def _traverse_len(self):
        return self._traverse_length

    def sample(self):
        if self.depth_uniform:
            return {self.key: random.choice(random.choice(self.depth_cands))}
        if self.ndim == 1:
            cands = list(range(self.start, self.end+1, self.step))
            return {self.key: random.choice(cands)}

        pivot = 0
        values = []
        for start, end, step in zip(self.start, self.end, self.step):
            start = max(start, pivot)
            cands = list(range(start, end+1, step))
            v = random.choice(cands)
            if self.ascending:
                pivot = v
            values.append(v)
        return {self.key: values}

    def traverse(self):
        if self.ndim == 1:
            cands = list(range(self.start, self.end+1, self.step))
            for v in cands:
                yield {self.key: v}
        else:
            candidates = []
            for start, end, step in zip(self.start, self.end, self.step):
                candidates.append(list(range(start, end+1, step)))
            candidates = itertools.product(*candidates)
            if self.ascending:
                for values in candidates:
                    valid = True
                    prev = 0
                    for v in values:
                        if v < prev:
                            valid = False
                            break
                        else:
                            prev = v
                    if valid:
                        yield {self.key: values}
                    else:
                        continue
            else:
                for values in candidates:
                    yield {self.key: values}
                    
    def enumeration(self):
        min_depth = np.sum(self.start)
        max_depth = np.sum(self.end)
        bin_num = (max_depth - min_depth) + 1
        depth_cands = [[] for _ in range(bin_num)]
        candidates = []
        
        for start, end, step in zip(self.start, self.end, self.step):
            candidates.append(list(range(start, end+1, step)))
        candidates = itertools.product(*candidates)
        
        for each_candidate in candidates:
            temp_length = np.sum(each_candidate)
            depth_cands[temp_length-min_depth].append(each_candidate)
        return depth_cands

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'key={self.key}, '
        format_string += f'start={self.start}, '
        format_string += f'end={self.end}, '
        format_string += f'step={self.step}, '
        format_string += f'mode={self._mode}, '
        format_string += ')'
        return format_string

    
@MODEL_SAMPLERS.register_module('candidate')
class CandidateModelSampler(BaseModelSampler):
    """ Range model sampler.
    Args:
        key (str): name of model space dimension
        candidates (list[list]): list of candidates

    """
    def __init__(self, key, candidates, mode='sample'):
        assert isinstance(candidates, Sequence)
        super(CandidateModelSampler, self).__init__(mode)
        self.key = key
        # infer ndim
        one_dim = True
        for c in candidates:
            if isinstance(c, Sequence):
                one_dim = False
        self.ndim = 1 if one_dim else 2

        if self.ndim == 2:
            rectified_cands = []
            for c in candidates:
                if not isinstance(c, Sequence):
                    rectified_cands.append([c])
                else:
                    rectified_cands.append(c)
            self.candidates = rectified_cands
            self._traverse_length = np.prod([len(c) for c in rectified_cands])
        else:
            self.candidates = candidates
            self._traverse_length = len(candidates)

    def _sample_len(self):
        return 1

    def _traverse_len(self):
        return self._traverse_length

    def sample(self):
        if self.ndim == 1:
            return {self.key: random.choice(self.candidates)}
        elif self.ndim == 2:
            values = []
            for cs in self.candidates:
                values.append(random.choice(cs))
            return {self.key: values}
        else:
            raise NotImplementedError

    def traverse(self):
        if self.ndim == 1:
            for v in self.candidates:
                yield {self.key: v}
        elif self.ndim == 2:
            comb = product(*candidates)
            for v in comb:
                yield {self.key: v}
        else:
            raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'key={self.key}, '
        format_string += f'candidates={self.candidates}, '
        format_string += f'mode={self._mode}, '
        format_string += ')'
        return format_string

@MODEL_SAMPLERS.register_module('flops')
class FlopsModelSampler(BaseModelSampler):
    """ Flops model sampler.
    Args:
        key (str): the json which record the input_size, architecture and flops
    Examples:
        dict(
            type='flops',
            key="/data/GAIA-seg/hubs/flops/flops.json",
            input_size="3,512,1024",
            bin_num=5
        )
    """
    def __init__(self, key, input_size, bin_num=5, mode='sample'):
        super(FlopsModelSampler, self).__init__(mode)
        self.key = key
        self.input_size = input_size
        if isinstance(self.input_size,int):
            self.input_size = f"3,{self.input_size},{self.input_size}"
        self.bin_num = bin_num
        self.ndim = 1 
        self.candidates = self.get_candidates()
        self._traverse_length = 0
        for each_candidate in self.candidates:
            self._traverse_length += len(each_candidate)

    def _sample_len(self):
        return 1

    def _traverse_len(self):
        return self._traverse_length

    def sample(self):
        return random.choice(random.choice(self.candidates))[1]
        
    def traverse(self):
        for each_candidates in self.candidates:
            for candidate in each_candidates:
                yield candidate[1]
        
    def get_candidates(self):
        flops_json_file = open(self.key,'r')
        #pdb.set_trace()
        candidates = []
        for each_row in flops_json_file:
            info_dict = eval(each_row.strip())
            temp_input_shape = info_dict["data"]["input_shape"]
            if isinstance(temp_input_shape, int):
                temp_input_shape = f"3,{temp_input_shape},{temp_input_shape}"
            if temp_input_shape != self.input_size:
                continue
            temp_candidate = []
            temp_candidate.append(info_dict["overhead"]["flops"])
            temp_dict = {}
            for key,value in unfold_dict(info_dict).items():
                if 'arch' in key:
                    temp_dict[key] = value
            temp_candidate.append(temp_dict)
            candidates.append(temp_candidate)

        candidates = sorted(candidates, key=lambda x:x[0])
        assert len(candidates) > 0, "This flops json file doesn't contain this input size"
        n = int(math.ceil(len(candidates) / float(self.bin_num)))
        candidates = [candidates[i:i + n] for i in range(0, len(candidates), n)]
        return candidates

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'key={self.key}, '
        format_string += f'candidates={self.candidates}, '
        format_string += f'mode={self._mode}, '
        format_string += ')'
        return format_string
