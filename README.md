# Man Overboard Object Tracking
This repository is a companion page for the following paper:
> Sanne Eeckhout, Pratik Phadte , Gokul Ramalingan, Hussein Hesham Hassan Abdelaziz Gad. 2024. Man Overboard Object Tracking. Image Processing and Computer Vision. University Twente.

It contains all the material required for replicating the study, including: stabilization, camera calibration, object tracking using a customized CSRT tracker and distance calculation to the object using horizon detection.

## Quick start

### Getting started

The packages used by this repository can be installed using ```pip install opencv-python numpy```  
The code can be run using the `buoyTracker.py` file in the [src](src/) folder.

### Results
The main result of the tracking model is the generated video, which can be seen below.



## Repository Structure
This is the root directory of the repository. The directory is structured as follows:

     .
     |
     |--- src/                             Source code used in the paper
            |
            |--- buoyTracker.py            Main file containing the tracker model.
            |--- depthCalculation.py       Main script to run the inference and track power metrics
            |--- stabilization.py          Stabilizes the image using Lucas-Kanade Optical Flow
            |--- motionModel.py            Creates a motion model for the tracker model, 
     |--- documentation/                   Contains the figures presented in the paper.
     |
     |--- data/                            Contains coordinates of the detected object, manual ground truth and detected distances.


## How to cite us
The scientific article describing design, execution, and main results of this study is available [here](https://github.com/s-eeckhout/ipcv-project/report.pdf).<br> 
If this study is helping your research, consider to cite it is as follows, thanks!

```
@article{,
  title={Man Overboard Object Tracking},
  author={Eeckhout, Sanne and Phadte, Pratik and Ramalingan, Gokul and Gad, Hussein Hesham Hassan Abdelaziz.},
  journal={Image Processing and Computer Vision},
  year={2024},
  publisher={University Twente}
}
```

## Repository license
[MIT license](https://opensource.org/licenses/MIT)
