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

# Prepare Makefiles based on OS
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    cp makefile.unix makefile
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp makefile.mac makefile
elif [[ "$OSTYPE" == "cygwin" ]]; then
    echo "WARNING: Not tested under cygwin"
    cp makefile.unix makefile
elif [[ "$OSTYPE" == "win32" ]]; then
    echo "Windows is not supported. Terminating."
    exit
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    echo "WARNING: Not tested under FreeBSD"
    cp makefile.unix makefile
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

# Go to cuda-convnet directory
cd ./libraries/cuda-convnet2

# Compile cuda-convnet2
./build.sh

# Go back to the main directory
cd "$cwd"
