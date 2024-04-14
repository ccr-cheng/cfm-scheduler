import argparse
import os
import time

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models import get_flow_model
from datasets import get_dataset
from utils import seed_all, load_config, get_optimizer, get_scheduler, count_parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    train_set = get_dataset(config.datasets.train)
    test_set = get_dataset(config.datasets.test)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False, num_workers=32)

    # Model
    print('Building model...')
    model = get_flow_model(config.model, config.scheduler, config.encoder).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
    global_step = 0

    def sample(n_sample, n_step, method='euler'):
        with torch.no_grad():
            model.eval()
            traj = model.sample(method, n_sample, n_step, args.device).clamp(0, 1)
            img = make_grid(traj, nrow=8, normalize=False, value_range=(0, 1))
            writer.add_image('sample', img, global_step)
        return traj


    assert args.resume is not None, 'Please specify the checkpoint to resume from.'

    traj = sample(config.n_sample, config.n_step, 'ode')
    import pdb; pdb.set_trace()
    print('Sampling finished!')
