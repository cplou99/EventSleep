# EventSleep

Welcome to the EventSleep code repository!

ArmsShake             |  Hands2Head           |  Head   |  LegsShake 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/cplou99/EventSleep/blob/main/Gifs/ArmsShake-pos_neg.gif)  |  ![](https://github.com/cplou99/EventSleep/blob/main/Gifs/Hands2Head-pos_neg.gif) |  ![](https://github.com/cplou99/EventSleep/blob/main/Gifs/Head-pos_neg.gif)  | ![](https://github.com/cplou99/EventSleep/blob/main/Gifs/Legs-pos_neg.gif)


Here you will find a collection of useful scripts and tools designed to work with the EventSleep dataset.

## Python Environment
To ensure a smooth setup, we have included a EventSleepEnv.yml file that lists all the necessary libraries 
and their versions. You can easily create a Python environment with the required dependencies, executing:

```ruby
 conda env create -f EventSleepEnv.yml
```

## Toy Dataset
Furthermore, we have included a folder named "Toy_Data", which contains a small subset of the original data. 
This subset enables users to run the scripts without the need to download the entire dataset. In this way, this
folder contains some isolated clips from a few trials. Most of the scripts include an option to specify if
you want to run it either with the Toy_Data folder or with the original Dataset. Please note that to obtain the 
expected and accurate results, it is necessary to have access to the original complete dataset which will be 
stored, upon acceptance, in Synapse or Zenodo platforms.

## Pre-trained models
Additionally, we include a subfolder named "Models" which contains the trained models for our main
approaches. It also serves as a repository for storing any new model trained using the scripts.


## Scripts
Here is a list of the scripts included in this folder:

   - data_tools.py: A script with a wide range of preprocessing and post-processing tools 	
     to work with EventSleep dataset (extract labels from folder names, labels dictionaries 
     to obtain labels names, resize frames, crop frames to focus on the 
     bed area, etc...).
     
   - events_to_frames.py: A script to enable the transformation of the event data into 	
     frames according the events frame based representation explained in the paper. As result,
     it generates a folder named "EventFrames" in which you will find the resulting frames saved 
     in .npy format.

   - render_recordings.py: A script for visualizing the content of the event data, 
     leveraging the frames representation and providing a synchronized view with the 
     infrared recordings and the fine-grained ground truth labels.
     
   - train_ResNet-E: A script to train ResNet-E model. As result, it  generates a folder with the date 
     as name that contains a checkpoint.pth with the snapshot of the trained model and a 
     train_details.json file with the details of the run. This folder will be saved at Models/Events 
     in the folder corresponding to the labels and configurations used to train.
     
   - train_ResNet-IR.py: A script to train ResNet-IR model. As result, it generates a folder with the 
     date as name that contains a checkpoint.pth with the snapshot of the trained model and a 
     train_details.json file with the details of the run. This folder will be saved at Models/Infrared 
     in the folder corresponding to the labels and configurations used to train.
     
   - test_ResNet-E.py: A script to test a trained ResNet-E model model. As input, you 
     must provide the checkpoint path stored in Models folder. As result, it prints and save in 
     the checkpoint parent folder the confussion matrixes per configuration and averaged. 
     Additionally, it saves a test_details.json file with the details of the run. 
     
   - test_ResNet-IR.py:  A script to test a trained ResNet-IR model from infrared frames. As input, you 
     must provide the checkpoint path stored in Models folder. As result, it prints and save in 
     the checkpoint parent folder the confussion matrixes per configuration and averaged. 
     Additionally, it saves a test_details.json file with the details of the run. 


We hope you find this repository helpful for your research and exploration of the EventSleep dataset.
