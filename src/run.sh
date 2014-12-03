#!/bin/bash

#
# Run this file to setup environment variables and run the main.py
#

# Export environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/2.4.5/lib64:$HOME/Libraries/libjpeg/lib:../libraries/cuda-convnet2/util:../libraries/cuda-convnet2/nvmatrix:../libraries/cuda-convnet2/cudaconv3
export PYTHONPATH=../libraries/cuda-convnet2/

srun --partition=long --gres=gpu:1 --constraint=K20 --mem=12000 python main.py --gpu 0 --save-path ../log $*
