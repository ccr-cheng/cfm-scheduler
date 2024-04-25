# Different Schedulers for Training Flow Matching Models

In this repo, we tested different flow matching schedulers on the final performance of the flow matching models.

To train the models, run the following command:

```bash
python train.py <config-file>.yml --savename <save-name>
```

For example, run the following code to train the model with the linear scheduler on CIFAR-10 dataset:

```bash
python main.py configs/cifar10_linear.yml --savename cifar10_linear
```

To test the models, run the following command:

```bash
python main.py <config-file>.yml --mode inf --savename <save-name> --resume <model-path>
```



# Dataset

- CIFAR-10 raw images