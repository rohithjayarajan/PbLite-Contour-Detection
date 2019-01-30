<h1 align="center"> PbLite Contour Detection </h1>
<p align="center">
<a href='https://github.com/rohithjayarajan/PbLite-Contour-Detection/blob/master/LICENSE'><img src='https://img.shields.io/badge/License-GPLv3-blue.svg'/></a>
</p>

---

## Overview

Boundary or contour detection is an important
problem in computer vision and is well studied. Other important
problems like hierarchical image segmentation can be reduced
to a contour detection problem. A version of the state-of-the-art
edge detection algorithm Pb-Lite is discussed in this project. The
Pb Lite approach causes a jump in the performance of boundary
detection by suppressing false positives in the textured regions
due to the coupling of multiscale local texture, brightness, and
color information with a globalization framework.


## Dependencies

- [OpenCV][reference-id-for-OpenCV]: An Open Source Computer Vision Library released under a BSD license.
A complete installation guide for OpenCV can be found [here][reference-id-for-here].

- numpy

- scipy

- matplotlib

- sklearn

- imutils

[reference-id-for-OpenCV]: https://opencv.org/
[reference-id-for-here]: https://docs.opencv.org/3.3.1/d7/d9f/tutorial_linux_install.html

## Standard install via command-line
```
git clone --recursive https://github.com/rohithjayarajan/PbLite-Contour-Detection.git
cd <path to repository>
```

Run program for 10 images in the BSDS Images folder: 
```
cd Phase1
python ./Code/Wrapper.py
```

A detailed report can be found in Report.pdf
