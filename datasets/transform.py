import copy

import torchvision.transforms as T

_TRANSFORM_DICT = {
    'center_crop': T.CenterCrop,
    'random_crop': T.RandomCrop,
    'resize': T.Resize,
    'to_tensor': T.ToTensor,
    'normalize': T.Normalize,
    'random_horizontal_flip': T.RandomHorizontalFlip,
    'random_vertical_flip': T.RandomVerticalFlip,
    'random_rotation': T.RandomRotation,
    'random_affine': T.RandomAffine,
    'random_resized_crop': T.RandomResizedCrop,
    'random_grayscale': T.RandomGrayscale,
    'random_perspective': T.RandomPerspective,
    'random_erasing': T.RandomErasing,
    'color_jitter': T.ColorJitter,
}


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return T.Compose(tfms)
