#!/bin/bash

virtualenv env2

source env2/bin/activate

pip3 install torch==1.12.1
pip3 install torchvision==0.13.1
pip3 install prettytable
pip3 install prettytables