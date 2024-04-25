
gpu_id="1"
config_file="configs/cifar10_linear.yml"
savename="test_linear"
resume="ckpt/cifar10_linear_100000.pt"

# gpu_id="2"
# config_file="configs/cifar10_cos.yml"
# savename="test_cos"
# resume="ckpt/cifar10_cos_100000.pt"

# gpu_id="3"
# config_file="configs/cifar10_exp.yml"
# savename="test_exp"
# resume="ckpt/cifar10_exp_100000.pt"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval_gen_images.py ${config_file} --mode inf --savename ${savename} --resume ${resume}