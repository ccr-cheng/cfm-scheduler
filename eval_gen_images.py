import argparse
import os

from tqdm import tqdm
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
import torchvision.utils as vutils

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
    # sampler = 'ode'
    # return_traj = False
    sampler = 'euler'
    return_traj = True
    # return_traj = False
    imgdir = f'{logdir}/gen_images-n_step-{config.n_step}-{sampler}'
    if return_traj:
        imgdir += '-withTraj'

    dirs = [logdir, imgdir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # writer = SummaryWriter(logdir)

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


    def sample(n_sample, n_step, method='euler', return_traj=False):
        with torch.no_grad():
            model.eval()

            if method == 'ode':
                traj = model.sample(method, n_sample, n_step, args.device).clamp(0, 1)
            elif method == 'euler':
                traj = model.sample(method, n_sample, n_step, args.device, return_traj=return_traj).clamp(0, 1)

            if return_traj:
                # only take 64 frames. take the frames evenly spaced.
                # n_traj = 64
                n_traj = 16
                step_size = n_step // n_traj + 1

                traj_sampled = traj[::step_size]
                remain = n_traj - len(traj_sampled)

                if remain > 0:
                    traj_sampled = torch.cat([traj_sampled, traj[-remain:]], dim=0)

                traj = traj_sampled  # first dim: n_steps

                # traj = traj[::(n_step // n_traj)]
                # img = make_grid(traj, nrow=8, normalize=False, value_range=(0, 1))
            # writer.add_image('sample', img, global_step)
        return traj


    assert args.resume is not None, 'Please specify the checkpoint to resume from.'

    # save image
    # total_img = 30000
    total_img = 100
    b = config.n_sample

    n_batches = total_img // b + 1

    i = 0
    for bi in tqdm(range(n_batches)):
        # traj = sample(config.n_sample, config.n_step, 'ode')
        traj = sample(config.n_sample, config.n_step, sampler, return_traj)
        for i in range(b):
            cur_i = bi * b + i

            if return_traj:
                traj_i = traj[:, i:i + 1]  # n_step, 1, C, H, W
                traj_i = traj_i.squeeze(1)  # n_step, C, H, W
                # traj_j = vutils.make_grid(traj_i, nrow=8)
                traj_j = vutils.make_grid(traj_i, nrow=len(traj_i))
                vutils.save_image(traj_j, f'{imgdir}/sample-{cur_i:04d}.png')
            else:
                vutils.save_image(traj[i:i + 1], f'{imgdir}/sample-{cur_i:04d}.png')

    # import pdb; pdb.set_trace()
    print('Sampling finished!')

# 1. visualization (ode+euler) + fid (ode).
# 2. sample with euler
#    - return_traj: with euler
