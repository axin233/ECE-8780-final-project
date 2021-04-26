# ECE-8780-final-project

This is the repository for the ECE-8780 final project.

## Data 
Our code was developed based on the [video](https://drive.google.com/drive/folders/1zbkJxemim7m4imcDkOlHFMJZmctD_Pv4?usp=sharing).

## Video demo
Here is our video processing result.
- Blue rectangle: Region of interest.
- Green curve: Needle detection results.
- Magenta curve: Thread detection results.

https://user-images.githubusercontent.com/59490151/116119401-68464500-a68c-11eb-9fbf-4f4341882545.mp4



## System Configuration
The code requires a system equiped with:
- OpenCV 4.3.0
- CUDA 9.2.88

## To run the code
make

./test


## File and folder description
There are two .cpp files in our top level directory segmentation.cpp and main.cpp, both are main file and contain entry point to our program:
- main.cpp: the main file does not include the segmentation algorithm.
- segmentation.cpp: the main file includes the segmentation algorithm.
- folders: as their names

The make file provided will compile with main.cpp by default. To use the segmentation algorithm, user needs to change main.cpp in the make file to segmentation.cpp.


The rest of the folder contains the other files needed to run the respective program (you can ignore the redundant main.cpp in each individual folder).

## IMPORTANT
Since Clemson Palmetto does not support any video capture or window UI related functions, our code is designed to work on a individual computer for the convenience of demo. You could easily run the code and see some demo images by compile and run the program. Or, add the video output option from opencv and get a full video output.

To do test on Clemson Palmetto is a bit tricky:
- 1. Remove any function that is video related, imshow, waitKey, and window functions.
- 2. The entry loop "while(1)" needs to be changed to a for loop that reads one image at a time.

In our report, we have captured the first 1200 frames from video and upload it to Palmetto for testing. The segmentation.cpp is set to process first 1200 image frames, which means it is designed for testing mainly (you could run the whole video by remove the obvious if statement within the while loop). 




