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
from .base_matcher import BaseMatcher


__all__ = ['GivenLabelsMatcher']


class GivenLabelsMatcher(BaseMatcher):
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
        super(GivenLabelsMatcher, self).__init__(source, target, dataset_name)

    def match(self):
        source_labels = self.source_mapping.labels
        target2source = {self.source_mapping.sep2uni(sl): sl for sl in source_labels}
        return target2source

    def __repr__(self):
        return (f"GivenLabelsMatcher("
                f"source={self.source},"
                f"target={self.target},"
                f"dataset_name={self.dataset_name},"
                f")")
