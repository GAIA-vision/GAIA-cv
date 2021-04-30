# standard lib
from collections.abc import Sequence


# make Sequence hashable, which enables indexing in pandas.DataFrames
def list2tuple(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def unfold_dict(folded_meta):
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


