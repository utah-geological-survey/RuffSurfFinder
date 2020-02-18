# Ruff Surf Finder
This is a series of functions written by Matthew Morriss as an intern at the Utah Geological Survey. The primary objective of this project was to measure *surface roughness* and determine whether:
 1. Roughness can be used to aid surficial mapping geologists in mapping landslides and 
 2. Whether or not roughness can actually be leveraged to map landslides in an automated fashion.

## Installation
The scripts and the executbale file associated with `Ruff Surf Finder` are all writte in Python. All code has been written and tested using python 3.7. The Executable `RuffSurf.exe` is a frozen version of the code that should run without Python installed locally on a windows machine. There is not currently a unix compatible version of this function.

The executable scripts (e.g. `Snow_basin_Roughness_Maps.py`) have been tested on a Windows machine, but should run without issue on Unix machines. 

Ancillary packages required for successful code run: 
1. Time
2. OS
3. numpy
4. matplotlib
5. rasterio
6. osgeo
7. pandas
8. numba
	
All code was excuted with the latest version of these packages as of 2/14/20



## Description
National Science Foundation Fellow, Matthew Morriss, spent 6 months in late 2019 into early 2020 developing a set of codes to measure the surface roughness of lidar data. The premise of the project was to assess whether or not surface roughness could 1) aid a geologist in mapping landslides or 2) create audomated maps of landslides.

The code, herein, are the culmination of this project. 

Scripts that measure surface roughness include:
* `STDS.py` - Standard deviation of slope
* `RMSH.py` - Root mean squared height
* `conv2_mexh_var.py` and `conv2_mexh2.py` - continuous wavelet transform
* `DCE_preprocess.py` and `DCE_eig_par.py` - Directional cosine eigen vector

For a guided coding experience, I recommend you start with:
* `Snow_basin_Roughness_Maps.py` - which will guide you through all of the aforementioned functions.
* `Snow_basin_landslide_maps.py` - which will guide you through the deterministic solution to mapping landslides in a semi-automated fashion.


There are also smaller dependency functions not described herein. I also developed a Discrete Fourier Transform function which, due to the slowness of performance, I ceased using and will include but have not supported since early October 2019. 

** Examples **
A digital elevation model of the area around Snowbasin Ski Area.
![alt text](https://github.com/utah-geological-survey/RuffSurfFinder/blob/master/SB_ls_map.jpg)

A surface roughness model of the same area
![alt text](https://github.com/utah-geological-survey/RuffSurfFinder/blob/master/Figuer_17_STDS_Map.jpg)
**Inputs**

All functions take .tif files exported from a GIS software. Importantly, these should be square or rectangular clipped pieces of a Digital Elevation Model. DEM must be **projected in UTM**.

### RuffSurf.exe
The executable `RuffSurf.exe` is a standalone app, which runs on windows, to make a map of surface roughness across a digital elevation model. The user can use a GUI to:
1. Select the .tif DEM file they want to create a roughness map from.
2. Upload the file
3. Choose the method desired to calculate surface roughness
4. Input the desired window size
5. Run that algorithm to output a .tif file of surface roughness across the ROI.

This executable only contains 3 methods of surface roughness:
* Root mean squared height
* standard deviation of slope
* directional cosine eigen vector.

If the user is interested in experimenting with the Continuous Wavelet Transform, I suggest they examine the `Snow_basin_Roughness_Maps.py` script for a tutorial.


## Credits
This project would not be possible without building on the work of many other talented programmers and Earth Scientists. 

Code written in Matlab was supplied by: Drs. Matteo Berti and Adam Booth. This original code was ported to Python and heavily modified. Most methods were parallelized for optimized execution time, bringing 11 hours down to 15 minutes, on the test machine with 8 cores. 

## References
Methods developed and tested for this project are described in detail in:
Berti, M., Corsini, A., and Daehne, A., 2013, Comparative analysis of surface roughness algorithms for the identification of active landslides: Geomorphology, v. 182, p. 1-18., doi: 10.1016/j.geomorph.2012.10.022.

And the references within that paper. 

When the Open File Report is published with the Utah Geological Survey, I will include a link here that.