#!/bin/bash

# Q1
python3 lab5.py Q1 --batch_size 32 --gpus 4 --outfile 4GPU_batchsize_32.txt
python3 lab5.py Q1 --batch_size 128 --gpus 4 --outfile 4GPU_batchsize_128.txt
python3 lab5.py Q1 --batch_size 512 --gpus 4 --outfile 4GPU_batchsize_512.txt

python3 lab5.py Q1 --batch_size 32 --gpus 2 --outfile 2GPU_batchsize_32.txt
python3 lab5.py Q1 --batch_size 128 --gpus 2 --outfile 2GPU_batchsize_128.txt
python3 lab5.py Q1 --batch_size 512 --gpus 2 --outfile 2GPU_batchsize_512.txt

python3 lab5.py Q1 --batch_size 32 --gpus 1 --outfile 1GPU_batchsize_32.txt
python3 lab5.py Q1 --batch_size 128 --gpus 1 --outfile 1GPU_batchsize_128.txt
python3 lab5.py Q1 --batch_size 512 --gpus 1 --outfile 1GPU_batchsize_512.txt