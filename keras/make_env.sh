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
# download opencv-2.4.11

git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0
mkdir release
cd release
# compile and install
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/mnt/opencvlib -D PYTHON2_EXECUTABLE=/usr/bin/python3 -D WITH_CUDA=OFF ..
make all -j4 # 4 cores
sudo make install
cd /usr/local/lib/python3.4/dist-packages
sudo ln -s /mnt/opencvlib/lib/python3.4/dist-packages/cv2.cpython-34m.so cv2.so
sudo ln /dev/null /dev/raw1394
sudo apt-get install python3-matplotlib python3-numpy python3-pil python3-scipy -y
sudo apt-get install build-essential cython -y
sudo apt-get install python3-skimage -y
sudo python3 -m pip install pyyaml
sudo python3 -m pip install git+git://github.com/jakebian/quiver.git
sudo python3 -m pip install keras
sudo python3 -m pip install Cython
cd /mnt/Bachelor-2017/keras
sudo python3 setup.py build_ext --inplace #build cypthon
sudo python3 -m pip install scikit-learn
sudo python3 -m pip install h5py

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

sudo python3 main.py 6 >> log_vgg16_ft_dataset_.txt && sudo python3 main.py 7 >> log_vgg16_ft_dataset_art.txt && sudo python3 main.py 8 >> log_vgg16_ft_dataset_rand.txt && sudo python3 main.py 9 >> log_vgg16_ft_dataset_seg.txt