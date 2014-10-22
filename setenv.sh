#!/bin/bash

#
# Run this file to setup environment variables and run the main.py
#

# Export environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/2.4.5/lib64:$HOME/Libraries/libjpeg/lib:$HOME/Deepmind/Replicating-DeepMind/libraries/cuda-convnet2/util:$HOME/Deepmind/Replicating-DeepMind/libraries/cuda-convnet2/nvmatrix:$HOME/Deepmind/Replicating-DeepMind/libraries/cuda-convnet2/cudaconv3
export PYTHONPATH=$HOME/Deepmind/Replicating-DeepMind/libraries/cuda-convnet2/