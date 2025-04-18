# CMSE802_project
# Title: 
Multi-category Image Classification of Finch Behaviors
# Description: 
A computer vision project utilizing  a CNN (MobileNet V2) for multi-category image classification of Zebra Finch behaviors. The overall goal is to be able to have the model classify the image of the finch as either exhibiting the behaviors of "perched", "In nest", "No birds present", "Entering", or "exiting".
# Objectives: 
1. Achieving relatively high classification accuracy on the test set.
2. Implement data augmentation techniques to artificially balance out classes (since the dataset is unbalanced)
3. Create a dataset of images extracted from the video frames captured in the lab 
# Instructions: 
This notebook requires the installation of additional Python packages. The packages can be installed via pip install [package_name] or via conda install -c [channel] [package_name]. Please install the latest version of each of the packages. The packages needed are os for operating system functions, shutil for file operations, cv2 for image processing, matplotlib and seaborn for plotting, re for regular expressions, random for shuffling and random number generation, PIL the python imaging library, tensorflow for the machine learning functions, sklearn for the model insights, numpy for numerical operations, and pandas for making dataframes. 
Other requirments include:
skimage for image augmentation
tqdm for progress visualization
glob for Unix-style pathfinding
multiprocessing for multi-core processing
# Folder Structure: 
EDA: contains files related to performing EDA and IDA; Dataset_creation: files related to the creation of the dataset; Classification: files related to performing classification and implementation of the model.; Model_Results: All files related to displaying the results and performance metrics after running the model.
