#!/usr/bin/env bash
sudo apt-get update && apt-get upgrade -y
sudo apt-get install gcc -y
sudo apt-get install git -y
sudo apt-get install make -y
sudo apt-get install linux-headers-$(uname -r) -y
sudo apt-get install libcupti-dev -y
sudo apt-get install dkms build-essential linux-headers-generic -y
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda -y
wget https://www.dropbox.com/home/libcudnn6_6.0.21-1%2Bcuda8.0_amd64.deb
sudo apt-get install python3-pip python3-dev -y
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp34-cp34m-linux_x86_64.whl
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils -y
sudo apt-get install python3.5-dev -y
sudo apt-get clean
sudo apt-get install python3-matplotlib python3-numpy python3-pil python3-scipy -y
sudo apt-get install build-essential cython -y
sudo apt-get install python3-skimage -y
sudo python3 -m pip install pyyaml
sudo python3 -m pip install git+git://github.com/jakebian/quiver.git
sudo python3 -m pip install Cython
cd /mnt/Bachelor-2017/keras
sudo python3 setup.py build_ext --inplace #build cypthon
sudo python3 -m pip install scikit-learn
sudo python3 -m pip install h5py
sudo python3 -m pip install psutil
