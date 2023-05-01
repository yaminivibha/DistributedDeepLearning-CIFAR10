#!/bin/bash

# Q1
python3 lab5.py Q1 --batch_size 32 --outfile batchsize_32 .txt
python3 lab5.py Q1 --batch_size 128 --outfile batchsize_128 .txt
python3 lab5.py Q1 --batch_size 512 --outfile batchsize_512 .txt

