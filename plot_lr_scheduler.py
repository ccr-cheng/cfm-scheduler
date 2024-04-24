import argparse
from models import get_flow_model
from utils import seed_all, load_config, get_optimizer, get_scheduler, count_parameters
from datasets import get_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models.schedulers import get_fm_scheduler



def get_scheduler_results(time_steps, config_dir):

    config = load_config(config_dir)
    scheduler = get_fm_scheduler(config.scheduler)
    seed_all(config.train.seed)

    lr_list = []
    deriv_list = []
    deriv_divide_list = []

    for t in time_steps:
        lr = scheduler.get_interpolant(t)
        lr_list.append(lr)

        deriv = scheduler.get_log_deriv(t)
        deriv_list.append(deriv)

        deriv_divide = deriv * (t - 1)
        deriv_divide_list.append(deriv_divide)
    return lr_list, deriv_list, deriv_divide_list



def plot_lr(time_steps, cos, exp, linear, savename):
    plt.plot(time_steps, cos, label='Cosine scheduler')
    plt.plot(time_steps, exp, label='Exponential scheduler')
    plt.plot(time_steps, linear, label='Linear scheduler')
    plt.xlabel('Time')
    plt.ylabel(f'{savename.replace("_", " ")}')

    y_min = min(min(cos), min(exp), min(linear))
    y_max = max(max(cos), max(exp), max(linear))
    y_min = max(y_min, -100)
    y_max = min(y_max, 100)
    plt.ylim(y_min, y_max)

    plt.title(f'{savename.replace("_", " ")} for different schedulers')
    plt.legend()
    plt.savefig(f'{savename}.png')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    time_steps = torch.arange(0, 0.99, 0.01)

    config_dir = 'configs/cifar10_cos.yml'
    cos_results = get_scheduler_results(time_steps, config_dir)

    config_dir = 'configs/cifar10_exp.yml'
    exp_results = get_scheduler_results(time_steps, config_dir)

    config_dir = 'configs/cifar10_linear.yml'
    linear_results = get_scheduler_results(time_steps, config_dir)

    time_steps = time_steps.numpy()
    plot_lr(time_steps, cos_results[0], exp_results[0], linear_results[0], 'schedule')
    plot_lr(time_steps, cos_results[1], exp_results[1], linear_results[1], 'derivative')
    plot_lr(time_steps, cos_results[2], exp_results[2], linear_results[2], 'loss_scale')