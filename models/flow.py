import torch
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

from .schedulers import Scheduler


class Flow(nn.Module):
    def __init__(self, encoder: nn.Module, scheduler: Scheduler, max_t=0.999, data_dims=(3, 32, 32)):
        """
        Conditional Flow Matching model
        :param encoder: the encoder network that takes t and xt as input and outputs the vector field
        :param scheduler: the scheduler κ(t) that controls the interpolation
        :param max_t: maximum timestep to avoid numerical instability
        :param data_dims: dimensions of the input data
        """
        super().__init__()
        self.encoder = encoder
        self.scheduler = scheduler
        self.max_t = max_t
        self.data_dims = data_dims

    def forward(self, t, xt):
        """
        predict vector field at time t
        :param t: timestep between 0 and 1, Tensor of shape (B,)
        :param xt: images, Tensor of shape (B, C, H, W)
        :return: predicted vector field, Tensor of shape (B, C, H, W)
        """
        if t.dim() == 0:
            t = t.expand(xt.size(0))
        vf = self.encoder(xt, t)
        return vf

    def get_loss(self, x):
        """
        get loss for training
        :param x: images, Tensor of shape (B, C, H, W)
        :return: a scalar loss
        """
        noise = torch.randn(*x.size(), device=x.device)
        t = torch.rand(x.size(0), device=x.device) * self.max_t
        interpolant_t = self.scheduler.get_interpolant(t)[:, None, None, None]
        xt = noise * interpolant_t + x * (1 - interpolant_t)
        vf = (xt - x) * self.scheduler.get_log_deriv(t)[:, None, None, None]
        pred_vf = self(t, xt)
        loss = (pred_vf - vf).pow(2).sum(-1).mean()
        return loss

    @torch.no_grad()
    def sample_euler(self, n_sample, n_step, device, return_traj=False):
        """
        sample from the model using Euler method
        :param n_sample: number of samples
        :param n_step: number of steps for Euler method
        :param device: device to run the model
        :param return_traj: whether to return the trajectory
        :return: samples, Tensor of shape (n_sample, C, H, W) or (n_step + 1, n_sample, C, H, W)
        """
        xs = [torch.randn(n_sample, *self.data_dims, device=device)]
        x = xs[0]
        ts = torch.linspace(0, 1, n_step + 1, device=device)
        dt = 1 / n_step
        for t in tqdm(ts[:-1], desc='Euler sampling'):
            pred_vf = self(torch.full((n_sample,), t, device=device), x)
            x = x + pred_vf * dt
            if return_traj:
                xs.append(x.detach())
        if return_traj:
            xs = torch.stack(xs, dim=0)
            return xs.detach()
        return x.detach()

    @torch.no_grad()
    def sample_ode(self, n_sample, n_step, device):
        """
        sample from the model using ODE solver
        :param n_sample: number of samples
        :param n_step: not used for ODE solver
        :param device: device to run the model
        :return: samples, Tensor of shape (n_sample, C, H, W)
        """
        print('Start ODE sampling')
        x0 = torch.randn(n_sample, *self.data_dims, device=device)
        x1 = odeint(
            self.forward,
            x0,
            t=torch.linspace(0, 1, 2, device=device, dtype=torch.float),
            atol=1e-5,
            rtol=1e-5,
        )[-1]
        print('End ODE sampling')
        return x1.detach()

    def sample(self, method, n_sample, n_step, device, **kwargs):
        assert method in ['euler', 'ode'], f'Unknown sampling method: {method}'
        if method == 'euler':
            res = self.sample_euler(n_sample, n_step, device, **kwargs)
        else:
            res = self.sample_ode(n_sample, n_step, device, **kwargs)
        return res
