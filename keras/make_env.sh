#!/usr/bin/env bash
apt-get update && apt-get upgrade -y
apt-get install gcc -y
apt-get install git -y
apt-get install make -y
apt-get install linux-headers-$(uname -r) -y
apt-get install libcupti-dev
sudo apt-get install dkms build-essential linux-headers-generic
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
wget https://www.dropbox.com/home/libcudnn6_6.0.21-1%2Bcuda8.0_amd64.deb
