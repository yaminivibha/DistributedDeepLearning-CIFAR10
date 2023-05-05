#!/bin/bash

# Q1, Q2, Q3
python3 lab5.py Q1 --batch_size 8 --gpus 4 --outfile 4GPU_batchsize_32.txt
python3 lab5.py Q1 --batch_size 32 --gpus 4 --outfile 4GPU_batchsize_128.txt
python3 lab5.py Q1 --batch_size 128 --gpus 4 --outfile 4GPU_batchsize_512.txt
python3 lab5.py Q1 --batch_size 512 --gpus 4 --outfile 4GPU_batchsize_2048.txt
python3 lab5.py Q1 --batch_size 2048 --gpus 4 --outfile 4GPU_batchsize_8192.txt


# python3 lab5.py Q1 --batch_size 16 --gpus 2 --outfile 2GPU_batchsize_32.txt
# # python3 lab5.py Q1 --batch_size 128 --gpus 2 --outfile 2GPU_batchsize_128.txt
# python3 lab5.py Q1 --batch_size 64 --gpus 2 --outfile 2GPU_batchsize_128.txt
# python3 lab5.py Q1 --batch_size 256 --gpus 2 --outfile 2GPU_batchsize_512.txt
# python3 lab5.py Q1 --batch_size 1024 --gpus 2 --outfile 2GPU_batchsize_2048.txt

# python3 lab5.py Q1 --batch_size 32 --gpus 1 --outfile 1GPU_batchsize_32.txt
# python3 lab5.py Q1 --batch_size 128 --gpus 1 --outfile 1GPU_batchsize_128.txt
# python3 lab5.py Q1 --batch_size 512 --gpus 1 --outfile 1GPU_batchsize_512.txt
# # python3 lab5.py Q1 --batch_size 1024 --gpus 1 --outfile 1GPU_batchsize_1024.txt
# python3 lab5.py Q1 --batch_size 2048 --gpus 1 --outfile 1GPU_batchsize_2048.txt
# python3 lab5.py Q1 --batch_size 8196 --gpus 1 --outfile 4GPU_batchsize_8192.txt

# Q4
