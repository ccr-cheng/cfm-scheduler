from abc import ABC, abstractmethod

import torch
import numpy as np


class Scheduler(ABC):
    """Base class for flow matching schedulers
    A scheduler κ(t) is defined such that κ(0) = 1, κ(1) ≈ 0
    """

    @abstractmethod
    def get_interpolant(self, t):
        """
        Get the value of the scheduler at time t: κ(t). Used for interpolation.
        :param t: timestep between 0 and 1
        :return: the interpolant timestep
        """

    @abstractmethod
    def get_log_deriv(self, t):
        """
        Get the log-derivative of the scheduler at time t: d/dt log κ(t). Used for loss calculation.
        :param t: timestep between 0 and 1
        :return: the log-derivative of the scheduler
        """


class LinearScheduler(Scheduler):
    """Linear scheduler κ(t) = 1 - t"""

    def get_interpolant(self, t):
        return 1 - t

    def get_log_deriv(self, t):
        return -1 / (1 - t)


class ExponentialScheduler(Scheduler):
    """Exponential scheduler κ(t) = e^{-a t}"""

    def __init__(self, a=5.):
        self.a = a

    def get_interpolant(self, t):
        return torch.exp(-self.a * t)

    def get_log_deriv(self, t):
        return torch.full_like(t, -self.a)


class CosineScheduler(Scheduler):
    """Cosine scheduler κ(t) = cos^2((t+s) / (1+s) * π/2)"""

    def __init__(self, s=0.008):
        self.s = s
        self.k0 = np.cos(s / (1 + s) * np.pi / 2) ** 2

    def get_interpolant(self, t):
        return torch.cos((t + self.s) / (1 + self.s) * np.pi / 2) ** 2 / self.k0

    def get_log_deriv(self, t):
        return -np.pi * torch.tan((t + self.s) / (1 + self.s) * np.pi / 2) / (1 + self.s)


_SCHEDULER_DICT = {
    'linear': LinearScheduler,
    'exponential': ExponentialScheduler,
    'cosine': CosineScheduler,
}


def get_fm_scheduler(cfg):
    s_cfg = cfg.copy()
    s_type = s_cfg.pop('type')
    return _SCHEDULER_DICT[s_type](**s_cfg)
