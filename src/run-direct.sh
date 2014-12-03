#!/bin/bash

#
# Run this file to setup environment variables and run the main.py
#

# Export environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/2.4.5/lib64:$HOME/libjpeg/lib:../libraries/cuda-convnet2/util:../libraries/cuda-convnet2/nvmatrix:../libraries/cuda-convnet2/cudaconv3
export PYTHONPATH=../libraries/cuda-convnet2/

python main.py --gpu 1 --save-path ../log $*
