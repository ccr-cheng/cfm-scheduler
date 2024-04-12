from .flow import Flow
from .schedulers import get_fm_scheduler
from .cnn import ConvNet


def get_flow_model(model_cfg, scheduler_cfg, encoder_cfg):
    """
    Build the categorical flow model.
    :param model_cfg: model configs passed to the flow model, type indicates the model type
    :param scheduler_cfg: scheduler configs passed to the Scheduler class
    :param encoder_cfg: encoder configs passed to the encoder model
    :return: the flow model
    """
    return Flow(ConvNet(**encoder_cfg), get_fm_scheduler(scheduler_cfg), **model_cfg)
