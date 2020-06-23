# AI Coach Assistant

An AI and computer vision based system for the automation of coaching feedback for Olympic weightlifting.

## Requirements

- OpenPose, python API
- Kivy
- OpenCV
- Python 3+

## How to use it

![alt text](https://github.com/RubenAMtz/ai-sports-assistant/blob/master/img/lift.gif)

- Configure OpenPose, so that the environment variables from the python API are available in your path ([openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python))
- From the terminal call python `pose_classification.py`
- Select and Load a video.
- Hit Play and watch the system detect the pose and grade it accordingly.
- Enjoy

## Action recognition

The available poses for detection are:

- Start
- Take-off
- Power position
- Extension
- Reception-snatch/clean
- Jerk

## Action assessment

The model will produce an output per frame, each output will be a value between 1 and 5, 1 being excellent and 5 being really bad.

## Limitations

You have to bear in mind that the models were trainined with a limited amount of data. We know there is a good source of videos online for training, however, we decided to run our own experiments to validate every aspect of the data. We will keep on growing the dataset as time goes on, to increase the accuracy of the results.

## TODO

- Automatic Barbell Tracking 
- Metrics derived from this tracking such as speed, path, etc.
- Perspective detection so that results are more accurate.

## Feedback

- Please do not hesitate to leave feed back or share with us your videos so we can keep growing the data base and potentially improve the results: rubenalvarez@pixail.ai
