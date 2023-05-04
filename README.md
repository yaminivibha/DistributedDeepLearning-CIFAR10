# DistributedDeepLearning-CIFAR10

Distributed deep learning [1] is the de facto approach to training neural networks at scale. Synchronous SGD (SSGD) has been the most widely used training algorithms among all the learning algorithms. In this lab, we are going to experiment with PyTorch’s DataParallel Module [2], which is PyTorch’s SSGD implementation across a number of GPUs on the same server. In particular, we re-use lab 2 code with default SGD solver and its hyper-parameter setup (e.g., learning rate and weight decay) and 2 num workers IO processes, running up to 4 GPUs with DataParallel Module.

## Setup

Create fresh VM with following requirements
- N1 CPU with highmene-2 (13GB)
- 4 T1 GPUs
- Deep Learning for Linux Boot Disk with CUDA 11.3 pre-installed

Install NVIDIA drivers (type `y`)

Create virtual environment
$ python3 -m venv env
$ source env/bin/activate
$ git clone https://github.com/yaminivibha/DistributedDeepLearning-CIFAR10.git
$ cd DistributedDeepLearning-CIFAR10
$ bash setup.sh

