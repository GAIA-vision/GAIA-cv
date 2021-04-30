# standard lib
import os
import json
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from pathlib import Path
import warnings

# 3rd party lib
import pandas as pd

# local lib
from .utils import unfold_dict, fold_dict


@pd.api.extensions.register_dataframe_accessor("ms_manager")
class ModelSpaceManager:
    '''A class that manages the model space
    '''
    def __init__(self, pd_obj):
        self._validate(pd_obj)
        self._obj = pd_obj
        # replace classmethod for instantiation
        self.dump = self._instance_dump

    @staticmethod
    def _validate(obj):
        reserved_tags = {'data', 'arch', 'metric', 'overhead'}
        has_tags = {k:False for k in reserved_tags}
        for col in obj.columns:
            for k in reserved_tags:
                if k in col:
                    has_tags[k] = True

        for k, v in has_tags.items():
            if not v:
                warnings.warn(f'Lack of information about `{k}`in model space, please check.')
        if not any(has_tags.values()):
            raise Exception(f'Must include information about at least one of the`{reserved_tags}`.')

    def apply_rule(self, rule):
        return rule(self._obj)

    def plot_distribution(self, **kwargs):
        raise NotImplementedError

    def pack(self, fold=True):
        '''Different from pandas.DataFrame.to_dict() in two ways:
        1. Column names are folded, e.g. `arch.neck.depth` to '{arch:{neck:{depth: ...}}}
        2. Activate `orient='list'` in pandas.DataFrame.to_dict()
        '''
        raw_dict = self._obj.to_dict(orient='list')
        new_dict = [dict(zip(raw_dict, v)) for v in zip(*raw_dict.values())]
        if fold:
            return [fold_dict(d) for d in new_dict]
        else:
            return [unfold_dict(d) for d in new_dict]

    def _instance_dump(self, filename):
        with open(filename, 'w') as fout:
            out = self._obj.ms_manager.pack()
            for d in out:
                fout.write(json.dumps(d) + '\n')

    @classmethod
    def dump(cls, pd_obj, filename):
        with open(filename, 'w') as fout:
            out = pd_obj.ms_manager.pack()
            for d in out:
                fout.write(json.dumps(d) + '\n')

    @classmethod
    def load(cls, input_, single_frame=False):
        if isinstance(input_, str):
            ext = os.path.splitext(input_)[-1]
            if ext == '.json':
                model_metas = []
                with open(input_, 'r') as fin:
                    for line in fin:
                        model_meta = json.loads(line.strip())
                        model_meta = unfold_dict(model_meta)
                        model_metas.append(model_meta)
                pd_obj = pd.DataFrame(model_metas)
                ModelSpaceManager._validate(pd_obj)
            else:
                raise Exception('got {}'.format(input_))
        elif isinstance(input_, dict):
            model_metas = unfold_dict(input_)
            if single_frame:
                model_metas = {k:[v] for k,v in model_metas.items()}
            pd_obj = pd.DataFrame(model_metas)
            ModelSpaceManager._validate(pd_obj)
        elif isinstance(input_, list):
            model_metas = [unfold_dict(d) for d in input_]
            pd_obj = pd.DataFrame(model_metas)
            ModelSpaceManager._validate(pd_obj)
        else:
            raise NotImplementedError(f'`{type(input_)}` not supported yet.')

        return pd_obj
