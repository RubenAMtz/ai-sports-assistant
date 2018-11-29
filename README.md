# AI Coach Assistant

An AI and computer vision based system for the automation of coaching feedback for Olympic weightlifting.

![alt text](https://github.com/RubenAMtz/ai-coach-assistant/blob/master/lift%20example%202d%20pose.gif "2d pose estimation")

## The following pipeline is implemented:

![alt text](https://github.com/RubenAMtz/ai-coach-assistant/blob/master/proposal.jpg "pipeline")

## 2D pose estimation:

For 2D pose estimation we will be using the strategy proposed by "Real-Time Multi-Person 2d pose estimation using part affinity fields" 
paper. The code can be found [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose).  
Multiple strategies for 3D pose estimation where tested, however, as the literature states and as the results showed, it is not ready for
industrial applications, it does not perform well under conditions with multiple people in the scene and lacks ability to reconstruct
"awkward" poses.

## Spatio-Temporal Information (yet to be implemented)

The strategy to be follow is defined by the following papers: [Link1](https://xbpeng.github.io/projects/SFV/2018_TOG_SFV.pdf). [Link2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5995496).

## Camera pose estimation

Problem statement: There are multiple camera poses within a scene, we propose to extract this information within a scene by training a model 
using synthetic data. This is the proposed pipeline:

![alt text](https://github.com/RubenAMtz/ai-coach-assistant/blob/master/synthetic%20pipeline.jpg "pipeline")

Now, for extracting the camera pose we need to first define a region of interest by means of classification. This tasks has been implemented
by using the tensorflow API (SDD + mobilenet architecture). As the size of the synthetic data created was low we implemented transfer
learning. Here are some of the results (we early stop the training process at 1.1 total loss):

![alt text](https://github.com/RubenAMtz/ai-coach-assistant/blob/master/classification%20test%20results.jpg "classification")

## Metric estimation

Data collection is been done by implementing a web page, this is the progress so far:

![alt text](https://github.com/RubenAMtz/ai-coach-assistant/blob/master/webpage.jpg "webpage")

# Install

For the barbell classification process
```
git clone https://github.com/RubenAMtz/ai-coach-assistant.git
```

* Download the synthetic data and place it inside the image folder.  [Data](https://drive.google.com/open?id=1BIylyeNjo6i1_bxdbw1g23p9ePesXV0Y)

* Run `python split.py`. Train and test folders will be created inside the images folder with their corresponding XML files.

# Training

Tensorflow API offers a set of different architectures, for the showed results we used SSD + mobilenet. Here are the steps taken for that:

* Install tensorflow API
* We have provided a folder that contains the basic structure needed to be integrated later to the tensorflow API folder structure.
* Go to the Detector folder.
* Copy the images folder into this folder.
* Run xml_to_csv.py
* Run generate_tfrecord.py
* Go to the tensorflow API folder and find ~/object_detection/legacy
* You will see a similar structure as the recently discussed.
* Copy what you have in your data folder into the data folder from the API.
* Copy your images folder into /legacy
* Copy barbell_detection.py into /object_detection from the API.
* Run barbell_detection.py 
* Stop the process whenever it plateaus. We haven't implemented metrics to evaluate the performance of the training yet. 

## More synthetic data

If you want to change the synthetic set and add more images, open untitled.blend (requires blender). Add a backgrounds folder at the root
and add hdr images (360 environment maps). Then run the script within blender (script_v2.py).
