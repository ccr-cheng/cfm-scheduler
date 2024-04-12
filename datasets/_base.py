from torchvision.datasets import CIFAR10

from .transform import get_transform

_DATASET_DICT = {
    'cifar10': CIFAR10,
}


def get_dataset(cfg):
    d_cfg = cfg.copy()
    d_type = d_cfg.pop('type')
    transform = get_transform(d_cfg.pop('transform', None))
    return _DATASET_DICT[d_type](**d_cfg, transform=transform)
