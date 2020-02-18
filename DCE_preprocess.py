# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:22:30 2019
DCE Preprocess, gets all of the inputs needed for the optimized DC_Eig_par function. This fucntion calculates the directional consines across the entire DEM (cy, cx, cz). This is a heavily modified function from Matteo Berti. 

This separate script was written as several of the function calls herein are not supported by Numba, which was used to parallelize the secondary function (DC_eig_par.py).

INPUT: 
    DEM - digital elevation model;
    cellsize - grid resolution
    w - size of moving window
    
OUTPUT:
    cy
    cx
    cz
        Directional cosine(s) calculated in the x, y, and z directions
    
EXAMPLES:
    w = 20
    cellsize = 0.5
    [cx,cy,cz] = DCE_preprocess(DEM,cellsize,w)
    
REFERENCES:
    McKean and Roering, 2004, Objective landslide detection and surface morphology mapping using high-resolution airborne laser altimetry, Geomorphology, v. 57, no. 3-4, 331 - 351.
    
@author: matthewmorriss
"""
import numpy as np
def DCE_preprocess(DEM, cellsize, w):
    
    [nrows, ncols] = np.shape(DEM)

    #initiate an empty array same size as dem
    rms = DEM*np.nan
#    rms = np.float32(rms)

#    #compute the directional cosines
    [fx, fy] = np.gradient(DEM, cellsize, cellsize)


    grad = np.sqrt(fx**2 + fy**2)
    asp = np.arctan2(fy, fx)



    grad=np.pi/2-np.arctan(grad) #normal of steepest slope
    asp[asp<np.pi]=asp[asp<np.pi]+[np.pi/2]
    asp[asp<0]=asp[asp<0]+[2*np.pi]

    #spherical to cartesian conversion
    r = 1
    cy = r * np.cos(grad) * np.sin(asp)
    cx = r * np.cos(grad) * np.cos(asp)
    cz = r * np.sin(grad)
    
    return(cx,cy,cz)