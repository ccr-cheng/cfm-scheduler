
gt_dir="/nfs/ycheng/proj/cfm-scheduler/CIFAR-10-images/train-all"
# gen_dir="/nfs/ycheng/proj/cfm-scheduler/logs/test_linear/gen_images"
# gen_dir="/nfs/ycheng/proj/cfm-scheduler/logs/test_exp/gen_images"
gen_dir="/nfs/ycheng/proj/cfm-scheduler/logs/test_cos/gen_images"


# linear: 24.8911
# exp: 49.8486
# cos: 27.9221

gpu_id="4"

python -m pytorch_fid ${gt_dir} ${gen_dir} --device cuda:${gpu_id}