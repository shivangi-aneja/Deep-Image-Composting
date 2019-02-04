# Advanced Deep Learning Practical Course : Realistic Composite Image Creation Using GANs

## 1. How To Train the model

```bash
usage: main.py [-h] [-d DATASET] [--data-dirpath DATA_DIRPATH]
               [--n-workers N_WORKERS] [--gpu GPU] [-rs RANDOM_SEED]
               [-dr DISCRIMINATOR] [-gr GENERATOR]  [-d_lr D_LR]  [-g_lr G_LR]
               [-b BATCH_SIZE] [-d_opt D_OPTIM]  [-g_opt G_OPTIM]
               [-m MODEL_NAME] [-e EPOCHS]   [-rl RECON_LOSS]
               [-tf TF_LOGS]  [-mp PLOT_MATPLOTLIB]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'big'} (default: big)
  --data-dirpath DATA_DIRPATH
                        directory for storing downloaded data (default: data/)
  --n-workers N_WORKERS
                        how many threads to use for I/O (default: 2)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
                        random seed for training (default: 1)
  --dr DISCRIMINATOR, --discriminator DISCRIMINATOR
                        discriminator architecture name, {'test_discriminator1', 'resnet', 'patch'}
                        (default: patch)
  --gr GENERATOR, --generator GENERATOR
                        generator architecture name, {'test_generator1', 'skip1', 'skip2', 'multi1', 'multi2'}
                        (default: skip2)
  -d_lr D_LR, --d_lr D_LR
                        discriminator learning rate (default: 0.0001)
  -g_lr G_LR, --g_lr G_LR
                        generator learning rate (default: 0.0001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 16)
  -d_opt D_OPTIM, --d_optim D_OPTIM
                        optimizer, {'adam', 'sgd', 'adagrad', 'rms_prop'} (default: adam)
  -g_opt G_OPTIM, --g_optim G_OPTIM
                        optimizer, {'adam', 'sgd', 'adagrad', 'rms_prop'} (default: adam)
  -m MODEL_NAME, --model_name MODEL_NAME
                        name of model (default: 'gan_model')
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default: 1000)
  -rl RECON_LOSS, --recon_loss RECON_LOSS
                        loss function, {'l1','l2'} (default: l1)
  -tf TF_LOGS, --tf_logs TF_LOGS
                        folder for tensorflow logging
  -mp PLOT_MATPLOTLIB, --plot_matplotlib PLOT_MATPLOTLIB
                        whether to plot matplotlib
```

### Sample Command for training
```bash
python main.py -b 5 --gpu 1 -d_lr 1e-7 -g_lr 1e-5 -m pix2pix_patch_hue_total -e 1000 -tf tf_logs/pix2pix_patch_hue_total -rl l1 -dr patch -gr skip2
```

## 2. How To generate results on the model

```bash
usage: main.py [-h] [-d DATASET] [--data-dirpath DATA_DIRPATH]
               [--n-workers N_WORKERS] [--gpu GPU] [-rs RANDOM_SEED]
               [-dr DISCRIMINATOR] [-gr GENERATOR]  [-d_lr D_LR]  [-g_lr G_LR]
               [-b BATCH_SIZE] [-d_opt D_OPTIM]  [-g_opt G_OPTIM]
               [-m MODEL_NAME] [-e EPOCHS]   [-rl RECON_LOSS]
               [-tf TF_LOGS]  [-mp PLOT_MATPLOTLIB]

optional arguments:
  -h, --help    show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'big'} (default: big)
  --data-dirpath DATA_DIRPATH
                        directory for storing downloaded data (default: data/)
  --n-workers N_WORKERS
                        how many threads to use for I/O (default: 2)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
                        random seed for training (default: 1)
  --dr DISCRIMINATOR, --discriminator DISCRIMINATOR
                        discriminator architecture name, {'test_discriminator1', 'resnet', 'patch'}
                        (default: patch)
  --gr GENERATOR, --generator GENERATOR
                        generator architecture name, {'test_generator1', 'skip1', 'skip2', 'multi1', 'multi2'}
                        (default: skip2)
  -d_lr D_LR, --d_lr D_LR
                        discriminator learning rate (default: 0.0001)
  -g_lr G_LR, --g_lr G_LR
                        generator learning rate (default: 0.0001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 16)
  -d_opt D_OPTIM, --d_optim D_OPTIM
                        optimizer, {'adam', 'sgd', 'adagrad', 'rms_prop'} (default: adam)
  -g_opt G_OPTIM, --g_optim G_OPTIM
                        optimizer, {'adam', 'sgd', 'adagrad', 'rms_prop'} (default: adam)
  -m MODEL_NAME, --model_name MODEL_NAME
                        name of model (default: 'gan_model')
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default: 1000)
  -rl RECON_LOSS, --recon_loss RECON_LOSS
                        loss function, {'l1','l2'} (default: l1)
  -tf TF_LOGS, --tf_logs TF_LOGS
                        folder for tensorflow logging
  -mp PLOT_MATPLOTLIB, --plot_matplotlib PLOT_MATPLOTLIB
                        whether to plot matplotlib

### Sample Command for testing
```bash
python evaluate_models.py -b 5 --gpu 1 -d_lr 1e-7 -g_lr 1e-5 -m pix2pix_patch_hue_total -rl l1 -dr patch -gr skip2
```

## Trained Models
No pretrained models were used. Everything is trained from scratch.
All the trained models can be downloaded [here](https://drive.google.com/file/d/1Fb9XrDYKtzJiysEi79dC_NZlsrgUr-9o/view?usp=sharing).

## Documents
The reports and presentations can be found in `docs` directory.


## Results
The dataset can be downloaded [here] ().
The generated results can be downloaded here [here]().