# standard lib
import os.path as osp
import warnings
import copy
import json
from collections import OrderedDict

# 3rd party lib
import numpy as np

# mmcv lib
import mmcv


class LabelMapping(object):
    ''' Label mapping of a seperate dataset to the unified label space.
    Args:
        pass
    '''
    def __init__(self, name, meta):
        self._dataset_name = name
        self._sep2name = meta['sep2name']
        self._name2sep = meta['name2sep']
        self._sep2uni = meta.get('sep2uni', None)
        # sorted
        labels = list(self._sep2name.keys())
        self._labels = sorted(labels, key=lambda x:int(x))
        self._names = [self._sep2name[l] for l in self._labels]

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def names(self):
        return self._names

    @property
    def labels(self):
        return self._labels

    def has_sep2uni(self):
        if self._sep2uni is not None:
            if len(self._sep2uni) > 0:
                return True
        return False

    def add_sep2uni(self, sep2uni):
        assert isinstance(sep2uni, dict), 'sep2uni must be dict'
        self._sep2uni= {int(k):int(v) for k,v in sep2uni.items()}

    def sep2uni(self, sep_label):
        assert self._sep2uni is not None, \
            'The labels of `{}` are not mapped yet.'.format(self._dataset_name)
        return self._sep2uni[sep_label]

    def sep2name(self, sep_label):
        return self._sep2name[sep_label]

    def name2sep(self, name):
        return self._name2sep[name]

    def todict(self):
        meta = {
            'sep2name': self._sep2name,
            'name2sep': self._name2sep,
        }
        if self._sep2uni is not None:
            meta['sep2uni'] = self._sep2uni

        lm_info = {
            'name': self._dataset_name,
            'meta': meta,
        }
        return lm_info

    def dump(self, filename, show_label=True):
        ext = osp.splitext(filename)[-1]
        if ext == '.csv':
            self.to_csv(filename, show_label)
        else:
            lm_info = self.todict()
            mmcv.dump(lm_info, filename)

    def to_csv(self, filename, show_label=True):
        fout = open(filename, 'w')
        if show_label:
            header = 'label,name\n'
            fout.write(header)
            for l, n in self._sep2name.items():
                fout.write(f'{l},{n}\n')
        else:
            header = 'name\n'
            fout.write(header)
            for n in self._names: # names sorted by labels
                fout.write(f'{n}\n')
        fout.close()

    @staticmethod
    def read_csv(filename, dataset_name=None):
        """Load dataset label mapping from csv.
        filaname: dataset_name.csv
        Text format: (csv head is required, uni column is optional)
            label, name, uni(optional)
            0, aaa, 27
            1, bbb, 44
        Or: (csv head is required, uni column is optional)
            name, uni(optional)
            aaa, 27
            bbb, 44
        label should start with 0 and be contineous.
        """
        sep2name = {}
        name2sep = {}
        sep2uni = {}
        if dataset_name is None:
            dataset_name = osp.splitext(osp.basename(filename))[0]

        with open(filename, 'r') as fin:
            header = None
            for count, l in enumerate(fin):
                if header is None:
                    header = l.strip()
                    header_sp= header.split(',')
                    name_idx = header_sp.index('name')
                    label_idx = header_sp.index('label') if 'label' in header_sp else None
                    uni_idx = header_sp.index('uni') if 'uni' in header_sp else None
                    continue
                sp = l.strip().split(',')
                if label_idx is not None:
                    label = int(sp[label_idx].strip())
                else:
                    label = count - 1
                if uni_idx is not None:
                    uni = int(sp[uni_idx].strip())
                    sep2uni[label] = uni
                name = sp[name_idx].strip()
                sep2name[label] = name
                name2sep[name] = label
        lm_info = {
            'name': dataset_name,
            'meta': {
                'sep2name': sep2name,
                'name2sep': name2sep,
            }
        }
        if uni_idx is not None:
            lm_info['meta']['sep2uni'] = sep2uni

        return LabelMapping(**lm_info)

    @staticmethod
    def load(filename, dataset_name=None):
        ext = osp.splitext(filename)[-1]
        if ext == '.csv':
            return LabelMapping.read_csv(filename, dataset_name)
        else:
            lm_info = mmcv.load(filename)

    def __repr__(self):
        lm_info = self.todict()
        meta = lm_info['meta']
        return (f"LabelMapping(name={self._dataset_name},"
                f"meta={meta})")


