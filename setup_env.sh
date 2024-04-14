### NOTE: run this if install for the first time
# git clone https://github.com/lab4d-org/lab4d.git --recursive

name="cfm"
conda create --name ${name} python=3.9 -y && conda activate ${name}

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install tqdm tensorboard torchdiffeq easydict