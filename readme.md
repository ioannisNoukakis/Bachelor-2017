# Keras Bias v1.0
## Introduction
This tool is for computing class activation mappings on a large scale and perform a bias metric on these heatmaps
as described in the .pdf. In order to use it effectively, it was designed in two modules: The folder 'keras' contains
everything needed to run the heavy CAMs and bias metric task on any GPU capable device. 
The folder 'notebook' contains a notebook called 'DataExploration' that allows one to see its model performance in terms
of (background attention) bias. It is recommended to run the main file of 'keras' with the following arguments: 0, 1, 2,
3 and 7 in order to get a bias metric result. Once you have the two saved numpy saved array files, transfer them
into your computer and use them in the 'DataExploration' notebook. You should have also two data sets: the vanilla one and one with black
background (refer to the examples: 'keras/dataset' and 'keras/dataset_black_bg').

## Environment setup
This guide covers the installation on a amazone web service g2.2xlarge Ubuntu 14.04 instance. 
In order to setup the environment, type the following commands:
```
ssh -X -i <path to key> ubuntu@<public ipv4 adress>
sudo -i
cd /mnt
apt update && apt upgrade -y
apt install git -y
git clone https://github.com/ioannisNoukakis/Bachelor-2017.git
cd /Bachelor-2017/keras
cp deploy_part1.sh /mnt/deploy_part1.sh
cp deploy_part2.sh /home/ubuntu/deploy_part2.sh
chmod +x /mnt/deploy_part1.sh
chmod +x /home/ubuntu/deploy_part2.sh
/mnt/deploy_part1.sh
```
Now go to https://developer.nvidia.com/cudnn and download cuDNN v5.1 for CUDA 8.0 into /home/ubuntu. You'll have to 
create an account at nvidia's site in order to make this download. Once it's done run the following commands: 
```
cd /home/ubuntu
./deploy_part2.sh
chown -R ubuntu /mnt
```
Now you should be set to run the experiment. 

## Command line tool user guide
Each argument is separated by a comma.
+ 0 : generates the art background data set and the random background data set.
+1 [args] : Fine-tunes VGG16 with the following arguments: *data set's path, path to save the trained model, 
'GAP_CAM' for a model with a global average pooling layer or 'DENSE' of a fully connected layer, 
number of epochs, max images that the program can load into memory (recommended 5'000), 
'1' to force image resizing to a 256 x 256 shape. '0' to leave images unchanged.*
+ 2 [args] : Generates heatmaps for each sample in the test data set with the following arguments: 
*input data set's path, model's path, path to save heatmaps, '1' for generate heatmaps for all classes '0' 
for only the predicted class, how much images it can process at a time (recommended 10),
 mode for heatmaps generation, 'cv2' or 'tf' (recommened 'tf').*

+ 3 [args] : Computes the bias metric with the following arguments:
*path to the heatmaps, number of thread to use.*

+ 4 [args] : Prints how much sample has a class in a data set with the following arguments: 
*data set's path, 'csv' for csv printing or 'tab' for python array printing or '-' for normal printing.* 

+ 5 [args] : Trains the first model with the following arguments:
*data set's path, max images that the program can load into memory (recommended 5'000).*

+ 6 [args] : Computes and colors a CAM for each class of a datset. Does not require the command 
*2* to be run. Uses the following arguments:
*model's path, path to save the generated image*

+ 7 [args] : Saves the results of the bias experiment into two files (that are serialized numpy arrays) 
with the following arguments:
*path to heatmaps folder, path to save the correct predicitons, path to save the wrongs predictions.*
