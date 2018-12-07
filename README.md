# adl4cv_practikum
Advanced Deep Learning Practical Course

## Trello Board
https://trello.com/b/PMPjHo7Z/adl4cv-practical

## Slack
https://adl4cv-tum.slack.com

## 1. Python Setup

Prerequisites:
- Unix system (Linux or MacOS)
- Python version 3
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text)

`which virtualenv`

to point to the installed location.

Also, installing with pip should work (the *virtualenv* executable should be added to your search path automatically):

`pip3 install virtualenv`

Execute
`virtualenv -p python3 --no-site-packages venv`

Basically, this installs a sandboxed Python in the directory `.venv`. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this *virtualenv* in a shell you have to first
activate it by calling:

`source venv/bin/activate`

To test whether your *virtualenv* activation has worked, call:

`which python`

This should now point to `.venv/bin/python`.

Installing required packages:

`pip3 install -r requirements.txt`

## 2. Tensorflow

### (Current release for CPU-only)
`pip install tensorflow`

`pip install tensorboard`

### GPU package for CUDA-enabled GPU cards
`pip install tensorflow-gpu`

`pip install tensorboard`

## 3. Pytorch

### Install pytorch (MAC CPU-only / Linux GPU Cuda 9.0)
`pip3 install torch torchvision`

### Install IPython
`pip install Ipython`

### Install Tensorboard for pytorch
`pip3 install tensorboard-pytorch --no-cache-dir`