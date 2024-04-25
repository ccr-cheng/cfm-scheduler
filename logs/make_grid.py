import os
import glob
import numpy as np

import imageio

import torch
import torchvision.utils as vutils

# schedule = 'test_cos'
schedule = 'test_exp'
# schedule = 'test_linear'
# dirname = 'gen_images-n_step-200'
dirname = 'gen_images-n_step-300'
# dirname = 'gen_images-n_step-300-euler'

tgt_dir = f'{schedule}/{dirname}'

# tgt_dir = 'test_cos/gen_images-n_step-200'
# tgt_dir = 'test_exp/gen_images-n_step-200'
# tgt_dir = 'test_linear/gen_images-n_step-200'

out_name = f'{schedule}/{schedule}-{dirname}.png'


n_imgs = 64
img_paths = glob.glob(os.path.join(tgt_dir, '**.png'), recursive=True)
img_paths = sorted(img_paths)
## random sample
img_paths = np.random.choice(img_paths, n_imgs)

# img_paths = img_paths[:n_imgs]

# read imgs and make a 8x8 grid
imgs = [imageio.imread(p) for p in img_paths]
imgs = [torch.from_numpy(img) for img in imgs]
imgs = torch.stack(imgs)
imgs = imgs.permute(0, 3, 1, 2).float() / 255.
img_grid = vutils.make_grid(imgs, nrow=8, padding=2, normalize=True)
vutils.save_image(img_grid, out_name)

# print(imgs.shape)



# import pdb; pdb.set_trace()