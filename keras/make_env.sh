#!/usr/bin/env bash
apt-get update && apt-get upgrade -y
apt-get install gcc
apt-get install git
apt-get install make
apt-get install linux-headers-$(uname -r)
cd $HOME
wget http://fr.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
chmod +x NVIDIA-Linux*
./NVIDIA-Linux*
