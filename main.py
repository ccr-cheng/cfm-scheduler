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


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for x, _ in train_loader:
                x = x.to(args.device)
                loss = model.get_loss(x)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate()
                    sample(config.n_sample, config.n_step, 'euler')
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate():
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.train.valid_max_batch or len(test_loader)
            for i, (x, _) in tqdm(enumerate(test_loader), total=total):
                if i >= total:
                    break
                x = x.to(args.device)
                loss = model.get_loss(x)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar('valid/loss', val_loss, global_step)
        print(f'Step {global_step} valid loss {val_loss:.6f}')
        return val_loss


    def sample(n_sample, n_step, method='euler'):
        with torch.no_grad():
            model.eval()
            traj = model.sample(method, n_sample, n_step, args.device).clamp(0, 1)
            img = make_grid(traj, nrow=8, normalize=False, value_range=(0, 1))
            writer.add_image('sample', img, global_step)
        return traj


    try:
        if args.mode == 'train':
            train()
            print('Training finished!')
        if args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')

        sample(config.n_sample, config.n_step, 'ode')
        print('Sampling finished!')
        time.sleep(5)
    except KeyboardInterrupt:
        print('Terminating...')
