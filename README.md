# Deep Image Compositing : Realistic Composite Image Creation Using GANs

This work has been accepted at 14th WiML Workshop, NeurIPS Conference 2019. Find the attached poster [here](https://www.academia.edu/40947406/Deep_Image_Compositing).

## 1. How To train the model

```
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
```
python main.py -b 5 --gpu 1 -d_lr 1e-7 -g_lr 1e-5 -m pix2pix_patch_hue_total -e 1000 -tf tf_logs/pix2pix_patch_hue_total -rl l1 -dr patch -gr skip2
```

## 2. How To generate results on the model

```
usage: evaluate_models.py [-h] [-d DATASET] [--data-dirpath DATA_DIRPATH]
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
```
### Sample Command for testing
```bash
python evaluate_models.py -b 5 --gpu 1 -d_lr 1e-7 -g_lr 1e-5 -m pix2pix_patch_hue_total -rl l1 -dr patch -gr skip2
```

## Trained Models
No pretrained models were used. Everything is trained from scratch.
The trained model with Patch GAN can be downloaded [here](https://drive.google.com/file/d/1ioigvoe34oFKFcxFg32gkJsbwvRCnBpJ/view?usp=sharing).

## Documents
The reports and presentations can be found in `docs` directory.


## Downloads
The dataset can be downloaded [here](https://drive.google.com/file/d/1VG6U_zw8dFPlreq5toAgzE6xD2uDkbFC/view?usp=sharing).
The generated results can be downloaded [here](https://drive.google.com/file/d/1IwQ1FiVxQBWDu1p2_bNlEr94Peo-pVYK/view?usp=sharing). The directory `pred` contains generated images, `gt` contains ground truth and `comp` contains composite images.
The `.npy` file for evaluation can be downloaded [here](https://drive.google.com/file/d/1pH0H0R29AWe9OkXplx0yEHyA0JiOIBhA/view?usp=sharing).


## Results
<p float="left">
  <img src="/images/comp_4.png" width="23%" />
  <img src="/images/ht_4.png" width="23%" />
  <img src="/images/pred_4.png" width="23%" />
  <img src="/images/gt_4.png" width="23%" />
</p>

<p float="left">
  <img src="/images/comp_36.png" width="23%" />
  <img src="/images/ht_36.png" width="23%" />
  <img src="/images/pred_36.png" width="23%" />
  <img src="/images/gt_36.png" width="23%" />
</p>

<p float="left">
  <img src="/images/comp_39.png" width="23%" />
  <img src="/images/ht_39.png" width="23%" />
  <img src="/images/pred_39.png" width="23%" />
  <img src="/images/gt_39.png" width="23%" />
</p>

<p float="left">
  <img src="/images/comp_121.png" width="23%" />
  <img src="/images/ht_121.png" width="23%" />
  <img src="/images/pred_121.png" width="23%" />
  <img src="/images/gt_121.png" width="23%" />
</p>

<p float="left">
  <img src="/images/comp_149.png" width="23%" />
  <img src="/images/ht_149.png" width="23%" />
  <img src="/images/pred_149.png" width="23%" />
  <img src="/images/gt_149.png" width="23%" />
</p>
<pre> |     Composite       |       Deep Image        |        Predicted(Ours)      |       Ground Truth        | </pre>
