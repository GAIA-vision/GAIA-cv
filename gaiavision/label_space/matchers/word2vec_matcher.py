# standard lib
import os.path as osp
import warnings
import copy
import json
from collections import OrderedDict

# 3rd party lib
import numpy as np
try:
    import gensim
    import gensim.downloader
except Exception as e:
    raise NotImplementedError("pip install gensim to enable Word2vecMatcher")

# local lib
from ..label_mapping import LabelMapping
from .base_matcher import BaseMatcher


__all__ = ['Word2vecMatcher']


class Word2vecMatcher(BaseMatcher):
    ''' Match source label space into target label space.
    The matching relation is depend on the word2vec similarity.

    USAGE:
    1. Call Word2vecMatcher and feed source and target CSV into it;
    2. Word2vecMatcher will output intermediate results, and it records the most similar target labels for each source label based on the name similarity;
    3. You should review the intermediate results and decide which recommended target is correct manually;
    4. You should make a CSV file in the format of GivenLabelsMatcher source input, then call GivenLabelsMatcher.
    
    To use word2vec, you could download model via gensim.downloader.load(model_name), or manually download model_name and __init__.py from https://github.com/RaRe-Technologies/gensim-data/releases then put they into ~/gensim-data/model_name/
    For more detail about word2vec refer to https://radimrehurek.com/gensim/models/word2vec.html


    Args:
        source(csv): mapping of source label
        CSV format: (csv head is required)
            label, name
            0, aaa
            1, bbb
        target(csv): mapping of target label
        CSV format: (csv head is required)
            label, name
            0, aaa
            1, bbb
        dataset_name: name of source dataset
        sim_model: the model used to compute word2vec
        sim_threshold: word2vec similarity
        num_recommend: number of recommend in the intermediate results for each source row
    '''
    def __init__(self, source, target, dataset_name,
                 sim_model="word2vec-google-news-300",
                 sim_threshold=0.5,
                 num_recommend=5,
        ):
        super(Word2vecMatcher, self).__init__(source, target, dataset_name)
        self.sim_model = sim_model
        print("loading word2vec model {}".format(self.sim_model))
        self.model = gensim.downloader.load(self.sim_model)
        self.sim_threshold = sim_threshold
        self.num_recommend = num_recommend

    def match(self):
        source_labels = self.source_mapping.labels
        source_names = self.source_mapping.names
        target_names = self.target_mapping.names

        similar_matrix = {}
        for i, sl in enumerate(source_labels):
            sn = self.source_mapping.sep2name(sl)
            sn = sn.lower()
            similar_matrix[sl] = np.zeros((len(target_names)))
            if sn not in self.model:
                continue
            for j, tn in enumerate(target_names):
                tn = tn.lower()
                if tn not in self.model:
                    similar_matrix[sl][j] = 0
                else:
                    similar_matrix[sl][j] = self.model.similarity(sn, tn)
        return similar_matrix

    def to_csv(self, filename):
        similar_matrix = self.match()
        source_labels = self.source_mapping.labels
        source_names = np.array(self.source_mapping.names)
        target_names = np.array(self.target_mapping.names)

        # write
        fout = open(filename, 'w')
        header = ['label', 'name'] + ['top{}##label##name##score'.format(idx+1) for idx in range(self.num_recommend)]
        ntags = len(header)
        header = ','.join(header) + '\n'
        fout.write(header)

        for sl in source_labels:
            sn = self.source_mapping.sep2name(sl)
            values = [str(sl), sn]
            similar_score = similar_matrix[sl]
            sort_idx = np.argsort(-similar_score)
            sort_score = similar_score[sort_idx]
            sort_name = target_names[sort_idx]
            for top_i in range(self.num_recommend):
                tn = sort_name[top_i]
                tl = self.target_mapping.name2sep(tn)
                ts = sort_score[top_i]
                values.extend(["{}##{}##{:.2f}".format(tl, tn, ts)])
            fout.write(','.join(values)+'\n')
        fout.close()

    def __repr__(self):
        return (f"Word2vecMatcher("
                f"source={self.source},"
                f"target={self.target},"
                f"dataset_name={self.dataset_name},"
                f"sim_model={self.sim_model},"
                f"sim_threshold={self.sim_threshold},"
                f"num_recommend={self.num_recommend},"
                f")")
