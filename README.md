# Behaviorial Cloning Project

Overview
---
In this project, we train a Neural Network to steer a car in a driving simulation. The aim is for the NN to complete a lap around the track without leaving the driving lane. 

There are only two things we tweaked, the model architecture, and the data we use for training. Therefore, the NN goes directly from the data (images) to the output (steering) without us adding any specific knowledge to it.

Preface
---
As we started this project, we consulted the relevant slack channel for it, where we found a number of suggestions that we incorporated into our model from the beginning. 
* We made sure that we converted BGR to RGB.
* We spent a good amount of time obtaining more data, and driving the opposite way on the track.
* We started with a correction factor of +-0.20 for the side cameras.
* In drive.py we included a steering multiplier of 1.1.

The Architecture
---
Our starting point was the Nvidia architecture, which we tweaked a little bit, by adding dropout layers and by changing some of the parameters. The result can be seen below: 


The Data
---
Udacity provided us with a starting dataset for the project, of around 24k images. We added to it some 60k images, bringing the total to 86k data points. 

![left](examples/center_2016_12_01_13_31_14_602.jpg)
![left](examples/right_2017_10_26_12_18_47_124.jpg)

![left](examples/before.png)
![left](examples/after.png)

