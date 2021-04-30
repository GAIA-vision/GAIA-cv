# standard lib
import functools
import os
import pickle
import sys

# 3rd party lib
import torch
import torch.distributed as dist


MASTER_RANK = 0

def _serialize_to_tensor(data, group=None):
    device = torch.cuda.current_device()

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def broadcast_object(obj, group=None):
    """ broadcast object that could be pickled
    """
    if dist.get_world_size() == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()

    dist.broadcast(numel, MASTER_RANK)

    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)

    dist.broadcast(serialized_tensor, MASTER_RANK)

    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj


if __name__ == '__main__':
    import time
    from mmcv.runner import init_dist, get_dist_info
    init_dist('slurm')

    rank = dist.get_rank()
    arch = {'rank': rank, 'cul': list(range(rank+3))}
    print(arch)
    time.sleep(20)
    sync_arch = broadcast_object(arch)
    sync_arch['rank'] = rank
    print(sync_arch)

