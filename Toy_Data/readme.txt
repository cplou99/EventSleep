
***************************************  EVENTSLEEP DATASET ************************************************
    	
Welcome to the EventSleep dataset repository!

In this repository, you will find detailed information about the EventSleep dataset, which includes 
recordings from two different cameras: Event Camera (DVXplorer camera, 640 x 480 resolution) and Infrared 
Camera (ELP HD Digital Camera, 6fps). The dataset is organized into two main folders: EventCamera and
Infrared, corresponding to the recordings from each camera.

We split data into training and test sets based on subject (S) IDs. 
	- Train Set: S01, S02, S03, S04, S05, S06, S07, S08, S10, S11.
	- Test Set: S09, S12, S13, S14.
Each subject was recorded under three different configurations which implies 42 different trials. Thus, 
trials were named according the subject and the configuration ID. For instance, subject09_config2 will 
refer to the trial of the subject09 under configuration2.

In this way, each main folder is further divided into three subfolders:

	- TRAIN: This folder contains the trials of the train subjects, trimmed into action clips. 
	The order and label of each clip are provided in the clip names. Specifically, infrared clips 
	are given in folders containing the frames saved as .png whether event clips are given in .npy files.
	
	- TEST: This folder contains the trials of the test subjects, also trimmed into action clips. 
	Similar to the train set, the order and label of each clip are provided in the clip names. Specifically, 
	infrared clips are given in folders containing the frames saved as .png whether event clips are 
	given in .npy files.

	- TEST_FULL_SEQUENCE: This folder contains the full data sequences of the test subjects together 
	with an extra "in the wild" trial where the subject's (S15) actions do not follow the predetermined 
	sequence. Infrared clips are provided in .mp4 format, and event recordings are stored in the original 
	.aedat4 format. Additionally, we provide the labels in a Labels.csv file. In the infrared scenario, 
	the start and end of each label sequence are assigned to frame numbers, while in the event data, 
	are associated with the global timestamp of the camera in miliseconds.

Please note that, up to now, THIS IS JUST A SUBSET OF THE ORIGINAL DATASET called "Toy_Data", which allows 
you to run the provided scripts without downloading the entire dataset. However, for accurate and comprehensive 
results, it is necessary to wait for the original complete dataset which will be stored, upon acceptance, in 
Synapse or Zenodo platforms.

We hope you find this repository helpful for your research and exploration of the EventSleep dataset.
