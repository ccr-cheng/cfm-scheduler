
gt_dir="/nfs/ycheng/proj/cfm-scheduler/CIFAR-10-images/train-all"
gen_dir="logs/test_linear"

gpu_id="4"

python -m pytorch_fid ${gt_dir} ${gen_dir} --device cuda:${gpu_id}