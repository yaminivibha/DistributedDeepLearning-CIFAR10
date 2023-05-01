#!/bin/bash

virtualenv env

source env/bin/activate

pip3 install torch
pip3 install torchvision
pip3 install matplotlib
pip3 install prettytable