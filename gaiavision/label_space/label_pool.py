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

# local lib
from .label_mapping import LabelMapping


class LabelPool(object):
    ''' Label pool denotes a unified label space with label mapping
    to each separate datasets.
    Args:
        uni2name(dict): mapping of unified label to name
        name2uni(dict): mapping of name to unified label
        datasets(dict): info of each datasets
    '''
    def __init__(self, uni2name, name2uni, datasets=None):
        self._datasets = {}
        if datasets is not None:
            for ds_name, ds_info in datasets.items():
                self._datasets[ds_name] = LabelMapping(ds_name, ds_info)
        self._uni2name = uni2name
        self._name2uni = name2uni
        self._num_cats = len(uni2name)
        labels = list(self._uni2name.keys())
        self._labels = sorted(labels, key=lambda x:int(x))
        self._names = [self._uni2name[l] for l in self._labels]

    def __len__(self):
        return self._num_cats

    @staticmethod
    def _validate(lp_obj):
        pass

    @property
    def names(self):
        return self._names

    @property
    def labels(self):
        return self._labels

    @property
    def dataset_names(self):
        return list(self._datasets.keys())

    def dataset_classes(self, dataset_name):
        return self._datasets[dataset_name].names

    def uni2name(self, uni_label):
        if isinstance(uni_label, int):
            uni_label = str(uni_label)
        return self._uni2name[uni_label]

    def name2uni(self, name):
        return self._name2uni[name]

    def get_label_mapping(self, ds_name):
        return self._datasets[ds_name]

    def add_label_mapping(self, label_mapping):
        ds_name = label_mapping.dataset_name
        self._datasets[ds_name] = label_mapping
        return self

    def todict(self):
        datasets = {k:v.todict()['meta'] for k,v in self._datasets.items()}
        lp_info = {
            'uni2name': self._uni2name,
            'name2uni': self._name2uni,
            'datasets': datasets,
        }
        return lp_info

    def dump(self, filename):
        ext = osp.splitext(filename)[-1]
        if ext == '.csv':
            self.to_csv(filename)
        else:
            lp_info = self.todict()
            mmcv.dump(lp_info, filename)

    @staticmethod
    def _parse_csv_header(header):
        reserved_keys = {'name', 'label'}
        datasets = {}
        sp = header.strip().split(',')
        for t in sp:
            if t in reserved_keys: # uni tags
                continue
            tsp = t.strip().split('.')[:2]
            dataset = datasets.setdefault(tsp[0], set())
            dataset.add(tsp[1])
        for dataset in datasets.values():
            assert dataset == reserved_keys, f"Expect `{name, label}`, got {dataset}"

        return datasets

    def to_csv(self, filename):
        fout = open(filename, 'w')
        header = ['label','name']
        for dname in self._datasets:
            header.append(f'{dname}.label')
            header.append(f'{dname}.name')
        ntags = len(header)
        header = ','.join(header) + '\n'
        fout.write(header)

        for uni, name in self._uni2name.items():
            values = [str(uni), name]
            for dname, dataset in self._datasets.items():
                sep_names = []
                sep_labels = []
                for s, u in dataset._sep2uni.items():
                    if u == uni:
                        sep_labels.append(s)
                        sep_names.append(dataset._sep2name[s])
                if len(sep_labels) == 0:
                    values.extend(['',''])
                elif len(sep_labels) == 1:
                    values.extend([str(sep_labels[0]), sep_names[0]])
                else:
                    sep_labels = [str(v) for v in sep_labels]
                    values.extend(['|'.join(sep_labels), '|'.join(sep_names)])
            fout.write(','.join(values)+'\n')
        fout.close()

    @staticmethod
    def read_csv(filename):
        """Load dataset label pool from csv.
        Text format:
            label, name, d1.label(optional), d1.name(optional), d2.label(optional), d2.name(optional)
            0, aaa, 13, aaa1, 23, aaa2
            1, bbb, 24, bbb1, 27, bbb2
            ...
        label should start with 0 and be contineous.
        """
        uni2name = {}
        name2uni = {}
        datasets = {}

        with open(filename, 'r') as fin:
            header = None
            dataset_tags = None
            for l in fin:
                if header is None:
                    header = l.strip()
                    header_sp = header.split(',')
                    uni_name_idx = header_sp.index('name')
                    uni_label_idx = header_sp.index('label')
                    dataset_tags = LabelPool._parse_csv_header(header)
                    continue
                sp = l.strip().split(',')
                uni = int(sp[uni_label_idx].strip())
                name = sp[uni_name_idx].strip()
                uni2name[uni] = name
                name2uni[name] = uni
                # parse dataset tags if any
                for dname in dataset_tags:
                    dataset = datasets.setdefault(dname, {})
                    label_idx = header_sp.index(f'{dname}.label')
                    name_idx = header_sp.index(f'{dname}.name')
                    if len(sp[label_idx]) > 0:
                        sep2name = dataset.setdefault('sep2name', {})
                        name2sep = dataset.setdefault('name2sep', {})
                        sep2uni = dataset.setdefault('sep2uni', {})
                        if '|' in sp[label_idx]:
                            seps = [int(v.strip()) for v in sp[label_idx].split('|')]
                            names = [v.strip() for v in sp[name_idx].split('|')]
                            for i in range(len(seps)):
                                sep2name[seps[i]] = names[i]
                                name2sep[name[i]] = seps[i]
                                sep2uni[seps[i]] = uni
                        else:
                            sep = int(sp[label_idx])
                            name = sp[name_idx]
                            sep2name[sep] = name
                            name2sep[name] = sep
                            sep2uni[sep] = uni

        return LabelPool(uni2name=uni2name, name2uni=name2uni, datasets=datasets)

    @staticmethod
    def from_label_mapping(label_mapping):
        """ Build a LabelPool instance from a LabelMapping instance.
        """
        lp_info = {}
        assert isinstance(label_mapping, LabelMapping)
        sep2uni = {int(k):i for i, k in enumerate(label_mapping.labels)}
        name2uni = {name:sep2uni[label_mapping._name2sep[name]] for name in label_mapping.names}
        uni2name = {uni:name for name,uni in name2uni.items()}
        label_mapping.add_sep2uni(sep2uni)
        lm_info = label_mapping.todict()

        lp_info['name2uni'] = name2uni
        lp_info['uni2name'] = uni2name
        lp_info['datasets'] = {lm_info['name']: lm_info['meta']}
        return LabelPool(**lp_info)

    @staticmethod
    def load(filename, from_label_mapping=False):
        ext = osp.splitext(filename)[-1]
        if ext == '.csv':
            if from_label_mapping:
                label_mapping = LabelMapping.load(filename)
                return LabelPool.from_label_mapping(label_mapping)
            else:
                return LabelPool.read_csv(filename)
        else:
            lp_info = mmcv.load(filename)
            return LabelPool(**lp_info)

    def __repr__(self):
        return (f"LabelPool(uni2name={self._uni2name},"
                f"name2uni={self._name2uni},"
                f"datasets={self._datasets})")


