#!/bin/bash

#
#   Run this file from the project root to install all required libraries:
#       * Compile ALE under             ../libraries/ale
#       * Compile cuda-convnet2 under   ../libraries/cuda-convnet2
#


# ---
# ALE
# ---

# Store current directory name and go in ALE directory
cwd=$(pwd)
cd ./libraries/ale

# Prepare Makefiles based on OS, choose makefile with no SDL(no GUI)
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    cp makefile_noGUI.unix makefile
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp makefile_noGUI.mac makefile
elif [[ "$OSTYPE" == "cygwin" ]]; then
    echo "WARNING: Not tested under cygwin"
    cp makefile_noGUI.unix makefile
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "Windows is not supported. Terminating."
    exit
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "WARNING: Not tested under FreeBSD"
    cp makefile_noGUI.unix makefile
else
    echo "Unknown OS. Terminating."
    exit
fi

# Compile ALE
make

# Go back to the main directory
cd "$cwd"

# Test ALE
./libraries/ale/ale -help


# -------------
# cuda-convnet2
# -------------

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/2.4.5/lib64:$HOME/libjpeg/lib:$HOME/cuda-convnet2/util:$HOME/cuda-convnet2/nvmatrix:$HOME/cuda-convnet2/cudaconv3" >> ~./bash_profile
echo "export PYTHONPATH=$HOME/cuda-convnet2" >>  ~./bash_profile

# Go to cuda-convnet directory
cd ./libraries/cuda-convnet2

# Compile cuda-convnet2
./build.sh

# Go back to the main directory
cd "$cwd"
