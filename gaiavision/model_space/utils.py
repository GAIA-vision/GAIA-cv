# standard lib
from collections.abc import Sequence


def list2tuple(v):
    ''' This function is to make Sequence hashable,
    which enables indexing in pandas.DataFrames.
    '''
    if isinstance(v, list):
        return tuple(v)
    return v


def is_folded(meta):
    ''' Check whether a dict is folded
    '''
    is_folded_flag = True
    assert isinstance(meta, dict)
    for k, v in meta.items():
        if isinstance(v, dict):
            is_folded_flag = False
            break
    return is_folded_flag


def unfold_dict(folded_meta):
    ''' Unfold a dict and turn all lists into tuples.
    It is safe even if the dict is already unfolded.
    '''
    unfolded_meta = {}
    for k, v in folded_meta.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    unfolded_kk = '.'.join((k, kk))
                    unfolded_meta[unfolded_kk] = list2tuple(vv)
                else:
                    unfolded_vv = unfold_dict(vv)
                    unfolded_vv = {'.'.join((k, kk, kkk)):list2tuple(vvv) \
                            for kkk,vvv in unfolded_vv.items()}
                    unfolded_meta.update(unfolded_vv)
        else:
            unfolded_meta[k] = list2tuple(v)
    return unfolded_meta


def fold_dict(unfolded_meta):
    folded_meta = {}
    for k, v in unfolded_meta.items():
        k = str(k)
        if len(k.split('.')) == 1:
            folded_meta[k] = list2tuple(v)
        else:
            sp = k.split('.')
            d = folded_meta
            for ik in sp[:-1]:
                d = d.setdefault(ik, {})
            d[sp[-1]] = list2tuple(v)
    return folded_meta


