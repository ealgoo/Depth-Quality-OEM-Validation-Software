<p align="center"><img src="res/test.png" /></p>
<p align="center"><img src="res/pass.png" /></p>

## Before Run CMake & build
* The CMakeLists.txt inside the pakcage assume you have copy the pre-built DLLs & Libs in the Release tab and copy to the root project folder.
* If you want to build the related dlls & libs by self, please modify the CMakeLists.txt.

## Build Environment
* LibRS 2.8.0 - Windows has pre-built dll inside while under linux, requires to build from source code.
* Windows RS2
* VisualStudio v14 2015, libraries built with v14, other than v14, please build from source code as needed.
	- For pre-built dlls and libs, please download them from release page and paste to the root directory.
* Ubuntu 16.04, OpenCV, Pkgconfig, X11, GTK3, pthread, RealSense required.
* CMake 2.8.3+

## Overview

D400 IQC is the testing tool checking device quality through simple, cross-platform UI. The tool offers:

* Checking depth fill rate and distance from three ROI on image, Center, Bottom-Right corner, Top-Right corner.
* Each ROI contains 1/9 of the whole image resolution.
* JSON file provide flexibility for customer to change the test criteria.
	- FillRatePassPercentage: result needs to lager than 97% to pass fill rate test.
    - MeanPassPercentage: result needs to be larger than 470 – (470*0.02) AND smaller than 470 + (470 * 0.02).
	- TestDistance(mm): testing distance, use laser to test as ground truth.
	- EdgeOffset: for upper-right and bottom-down area, the offset enable the flexibility to reduce the right area in these two ROIs.
* Testing result will be shown at the left side panel with "Pass" or "Fail" also the test result.
* Testing result will also be saved inside the Result folder which located along with the IQC binary.
* Saved result include csv file, depth image and Left/Right IR image.

## Implementation Notes

You can get D400 IQC in form of a binary package on Windows and Linux, or build it from source alongside the rest of the library. The IQC tool is designed to be lightweight, requiring only a handful of embeded dependencies. Cross-platform UI is a combination of raw OpenGL calls, GLFW for cross-platform window and event management, and IMGUI for the interface elements. Please see COPYING for full list of attributions.