# Human-Pose-Estimation---AI-Development-Course
This repository contains the implementation of human pose estimation using the High-Resolution Network (HRNet). The HRNet architecture is designed to maintain high-resolution representations throughout the network, providing accurate keypoint detection even in complex and cluttered scenes.


## How to run?

### STEPS:

clone the repository

```bash
https://github.com/NaghamXD/Human-Pose-Estimation---AI-Development-Course
```

### STEP 01- Create a conda environment after opening the repository

```bash
micromamba create -n Hpe -c conda-forge python=3.12
```

```bash
micromamba activate Hpe
```

### STEP 02- Install the requirements

```bash
pip install -r requirements.txt
```

## Data

This is a set of 10,000 images gathered from Flickr searches for the tags 'parkour', 'gymnastics', and 'athletics'. Each image has
a corresponding annotation gathered from Amazon Mechanical Turk.
The images have been scaled such that the annotated person is roughly 150 pixels in length.

* README.md - this document
* joints.mat - a MATLAB format matrix 'joints' consisting of 14 joint locations and visibility flags. Joints are labelled in the following order:

    * Right ankle
    * Right knee
    * Right hip
    * Left hip
    * Left knee
    * Left ankle
    * Right wrist
    * Right elbow
    * Right shoulder
    * Left shoulder
    * Left elbow
    * Left wrist
    * Neck
    * Head top


* images/ - 10,000 images

Data Source:
Sam Johnson and Mark Everingham 
"Learning Effective Human Pose Estimation from Inaccurate Annotation" 
In proceedings of Computer Vision and Pattern Recognition (CVPR) 2011
@inproceedings{Johnson11, 
title = {Learning Effective Human Pose Estimation from Inaccurate Annotation}, 
author = {Johnson, Sam and Everingham, Mark}, year = {2011}, 
booktitle = {Proceedings of Computer Vision and Pattern Recognition (CVPR) 2011} }
E-mail: s.a.johnson04@leeds.ac.uk