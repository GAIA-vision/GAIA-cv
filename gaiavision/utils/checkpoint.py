# standard lib
import io
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

# 3rd-parth lib
import torch
import torchvision
from torch.optim import Optimizer
from torch.utils import model_zoo

# mm lib
import mmcv
from mmcv.runner.checkpoint import *


def load_checkpoint_with_surgeon(model,
                    filename,
                    surgeon,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI with model surgeons

    Args:
        model (Module): Module to load checkpoint.
        surgeons (T or list[T]): List of model surgeon that operate on state_dict
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # let surgeons operate on the state_dict
    if isinstance(surgeon, list):
        for surg in surgeons:
            state_dict = surg.operate_on(state_dict)
    else:
        state_dict = surgeon.operate_on(state_dict)

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

