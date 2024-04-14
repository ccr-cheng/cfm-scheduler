
gpu_id="2"
config_file="configs/cifar10_linear.yml"
savename="test_linear"
resume="ckpt/cifar10_linear_100000.pt"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval_fid.py ${config_file} --mode inf --savename ${savename} --resume ${resume}