#!/usr/bin/env bash
##After cudnn installation
sudo tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
export CUDA_HOME=/usr/local/cuda
sudo rm /etc/ld.so.conf.d/cuda-8-0.conf
sudo touch /etc/ld.so.conf.d/cuda-8-0.conf
sudo /usr/local/cuda/lib64 \/usr/local/cuda/extras/CUPTI/lib64 >> /etc/ld.so.conf.d/cuda-8-0.conf
sudo ldconfig
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/