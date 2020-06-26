# Traffic Signal Violation Detection System using Computer Vision

> <a href="https://youtu.be/PxHkKLUk_QM"><h3>Project Video Demonstration</h3></a>

> <a href="https://drive.google.com/open?id=1H5RjHPJ0CK5yq2_3vwwo4CwZt1NXTeVB"><h3>Project Report</h3></a>

> <a href="https://drive.google.com/open?id=1l46H2pNTbLLcIFSHWsMtor-vlvNSA4Gu"><h3>Project Slide</h3></a>


![Violation_Detection_Frame](Images/Violation_Detection_Frame_Red.jpg)

## INTRO
<img src="Images/Workflow Diagram.jpg" alt="Workflow Diagram" align="right" width="250" />
This is a software for practice of developing a system from completely scratch. Understanding this will help a lot in system development and basic structure of a system along with computer vision, GUI with python library Tkinter and basic opencv.

Go [here](#quick-starting-the-project) if you don't have time.

## Table of content

- [Motivation](#motivation)
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Quick Starting the project](#quick-starting-the-project)
- [System Overview](#system-overview)
- [Methodology](#methodology)
    - [Vehicle Classification](#vehicle-classification)
    - [Violation Detection](#violation-detection)
- [Implementation](#implementation)
    - [Computer Vision](#computer-vision)
    - [Graphical User Interface](#graphical-user-interface)
- [Contributing](#contributing)
- [Links and References](#links-and-references)
- [Author](#author)
- [Licensing](#licensing)

## Motivation
This project is made for the third year second semester System Development (CSE-3200) course.

## Introduction
The increasing number of cars in cities can cause high volume of traffic, and implies that traffic violations become more critical nowadays in Bangladesh and also around the world. This causes severe destruction of property and more accidents that may endanger the lives of the people. To solve the alarming problem and prevent such unfathomable consequences, traffic violation detection systems are needed. For which the system enforces proper traffic regulations at all times, and apprehend those who does not comply. A traffic violation detection system must be realized in real-time as the authorities track the roads all the time. Hence, traffic enforcers will not only be at ease in implementing safe roads accurately, but also efficiently; as the traffic detection system detects violations faster than humans. This system can detect traffic light violation in real-time. A user friendly graphical interface is associated with the system to make it simple for the user to operate the system, monitor traffic and take action against the violations of traffic rules.


## Objectives
The goal of the project is to automate the traffic signal violation detection system and make it easy for the traffic police department to monitor the traffic and take action against the violated vehicle owner in a fast and efficient way. Detecting and tracking the vehicle and their activities accurately is the main priority of the system. 

## Quick starting the project
1. `git clone https://github.com/anmspro/Traffic-Signal-Violation-Detection-System.git`
2. Download "yolov3.weights" from the link given in "yolov3 weights.txt"
3. Change the directories in "Project-GUI.py" & "object_detection.py"
3. Install required python dependencies into your python virtual environment.
4. `python Project-GUI.py`

## System Overview

![System Overview](Images/System_Flowchart.jpg)
            
     Figure-1: Flow diagram of traffic signal violation detection system.
The System consists of two main components -
* Vehicle detection model
* A graphical user interface (GUI)

First the video footage from the road side is sent to the system. Vehicles are detected from the footage. Tracking the activity of vehicles, system determines if there is any violation or not. Figure 1 shows how the system works.

The Graphical User Interface (GUI) makes the system interactive for the user to use. User can monitor the traffic footage and get the alert of violation with the detected bounding box of vehicle. User can take further action using the GUI. 

## Methodology
### Vehicle Classification
From the given video footage, moving objects are detected. An object detection model YOLOv3 is used to classify those moving objects into respective classes. YOLOv3 is the third object detection algorithm in YOLO (You Only Look Once) family. It improved the accuracy with many tricks and is more capable of detecting objects. The classifier model is built with Darknet-53 architecture. Table-1 shows how the neural network architecture is designed. 

![Darknet Architecture](Images/Darknet-53.png)


### Violation detection

The vehicles are detected using YOLOv3 model. After detecting the vehicles, violation cases are checked. A traffic line is drawn over the road in the preview of the given video footage by the user. The line specifies that the traffic light is red. Violation happens if any vehicle crosses the traffic line in red state.

The detected objects have a green bounding box. If any vehicle passes the traffic light in red state, violation happens. After detecting violation, the bounding box around the vehicle becomes red.


## Implementation
### Computer Vision
OpenCV is an open source computer vision and machine learning software library which is used in this project for image processing purpose. Tensorflow is used for implementing the vehicle classifier with darknet-53. 


### Graphical User Interface (GUI)
The graphical user interface has all the options needed for the software. The software serves administration and other debugging purposes. We don’t need to edit code for any management. For example, if we need to open any video footage, we can do it with the Open item (Figure-2). 

![Figure 2](Images/Initial_View.jpg)

     Figure-2: Initial user interface view.

Primarily, for the start of the project usage, the administrator needs to open a video footage using ‘Open’ item that can be found under ‘File’ (Figure-2). The administrator can open any video footage from the storage files (Figure-3).

![Figure 3](Images/Open_Video.JPG)

     Figure-3: Opening a video footage from storage.

After opening a video footage from storage, the system will get a preview of the footage. The preview contains a frame from the given video footage. The preview is used to identify roads and draw a traffic line over the road. The traffic line drawn by administrator will act as a traffic signal line. To enable the line drawing feature, we need to select ‘Region of interest’ item from the ‘Analyze’ option (Figure-4). After that administrator will need to select two points to draw a line that specifies traffic signal.

![Figure 4](Images/Select_Region_of_Interest.jpg)

     Figure-4: Region of Interest (Drawing signal line)

Selecting the region of interest will start violation detection system. The coordinates of the line drawn will be shown on console (Figure-5). The violation detection system will start immediately after the line is drawn. At first the weights will be loaded. Then the system will detect objects and check for violations. The output will be shown frame by frame from the GUI (Figure-6). 

![Figure 5](Images/Line_Coordinates.JPG)

     Figure-5: Line Coordinates (from console)

![Figure 6](Images/Violation_Detection_Frame.jpg)

     Figure-6: Final Output (on each frame)

The system will show output until the last frame of the footage. In background a ‘output.mp4’ will be generated. The file will be in ‘output’ folder of ‘Resources’. The process will be immediately terminated by clicking ‘q’.

After processing a video footage, the administrator can add another video footage from the initial file manager (Figure-2). If the work is complete the administrator can quit using ‘Exit’ item from File option.


Libraries used for graphical user interface:
* Tkinter


## Contributing
The main reason to publish something open source, is that anyone can just jump in and start contributing to my project.
So If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Links and References
- Project homepage: https://github.com/anmspro/Traffic-Signal-Violation-Detection-System
- Repository: https://github.com/anmspro/Traffic-Signal-Violation-Detection-System.git
- Issue tracker: https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/issues
- G. Ou, Y. Gao and Y. Liu, "Real TimeVehicularTrafficViolationDetectioninTrafficMonitorin gStream," in 2012 IEEE/WIC/ACM , Beijing, China , 2012.
- X. Wang, L.-M. Meng, B. Zhang, J. Lu and K.-L.Du, "A Video-based Traffic Violation Detection System," in MEC, Shenyang, China, 2013.
- Joseph Redmon and Ali Farhadi, "YOLOv3: An Incremental Improvement".
- https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras
- In case of any help you may need from me, please contact abunomanmd.sakib@gmail.com directly without any hesitation! I will be glad to help you.

## Author
Abu Noman Md. Sakib, Pias Roy<br>
abunomanmd.sakib@gmail.com<br>
pias.kuet@gmail.com<br>
Student at Department of Computer Science and Engineering<br>
Khulna University of Engineering & Technology, Khulna<br>
Bangladesh

<b> Supervised under </b><br>
Mahmudul Hasan Shauqi<br>
mahmudulhasanshauqi@gmail.com<br>
Lecturer<br>
Dept. of Computer Science and Engineering<br>
Khulna University of Engineering & Technology<br>

## Licensing
The code in this project is licensed under GNU GPLv3 license.

## Stargazers over time

![Stargazers over time](https://starchart.cc/anmspro/Traffic-Signal-Violation-Detection-System.svg)
      
