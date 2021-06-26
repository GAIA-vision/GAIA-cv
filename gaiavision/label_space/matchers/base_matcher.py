# standard lib
import os.path as osp
import warnings
import copy
import json
from collections import OrderedDict

# 3rd party lib
import numpy as np

# local lib
from ..label_mapping import LabelMapping


__all__ = ['BaseMatcher']


class BaseMatcher(object):
    ''' Match source label space into target label space.
    The matching relation is defined in the source csv.
    Every row in source csv consists of the label and name in source label space,
    and its corresponding label in target label space.

    Args:
        source(csv): mapping of source label
        CSV format: (csv head is required)
            label, name, uni
            0, aaa, 27
            1, bbb, 44
        target(csv): mapping of target label
        CSV format: (csv head is required)
            label, name
            0, aaa
            1, bbb
        dataset_name: name of source dataset
    '''
    def __init__(self, source, target, dataset_name):
        self.source = source
        self.target = target
        self.dataset_name = dataset_name
        self.source_mapping = LabelMapping.read_csv(self.source)
        self.target_mapping = LabelMapping.read_csv(self.target)

    def match(self):
        raise NotImplementedError

    def to_csv(self, filename):
        target2source = self.match()
        target_labels = self.target_mapping.labels

        # write
        fout = open(filename, 'w')
        header = ['label', 'name', '{}.label'.format(self.dataset_name), '{}.name'.format(self.dataset_name)]
        ntags = len(header)
        header = ','.join(header) + '\n'
        fout.write(header)

        for tl in target_labels:
            values = [str(tl), self.target_mapping.sep2name(tl)]
            if tl in target2source:
                sl = target2source[tl]
                sn = self.source_mapping.sep2name(sl)
                values.extend([str(sl), sn])
            fout.write(','.join(values)+'\n')
        fout.close()

    def __repr__(self):
        return (f"BaseMatcher("
                f"source={self.source},"
                f"target={self.target},"
                f"dataset_name={self.dataset_name},"
                f")")
