# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:32:17 2019
This is a function to calculate the standard deviation of slope (in parallel). Code modifeid from Matteo Berti (see github page for full reference). This code was written to take advantage of Numba's ability to process in parallel, vastly increasing the efficiency of the code.

INPUTS:
    DEM - a square shaped digital elevation model imported into python
    w - window size over which the STDS will be calculated
    cellsize - the grid spacing of the DEM
    
OUTPUTS:
    stdsGrid - the grid of calculated standard deviation of slopes.
    
@author: Matthew Morriss
"""
import numpy.core
import numpy as np
import numba as nb

@nb.njit(parallel=True)
def Slope_grd(DEM, w, cellsize):
    slope = np.nan * DEM
    [nrows, ncols] = np.shape(DEM)
    
    w1 = 2
    for i in nb.prange(w1,nrows):
        
        for j in range(w1,ncols):
            d1 = np.arange(i-w1,i+w1)
            d2 = np.arange(j-w1,j+w1)
            d3 = DEM[d1[0]:d1[-1],d2[0]:d2[-1]]
            
            dzx = (np.sum(d3[:,2])+d3[1,2] - np.sum(d3[:,0])-d3[1,0])/(8*cellsize)
            dzy = (np.sum(d3[0,:])+d3[0,1] - np.sum(d3[2,:])-d3[2,1])/(8*cellsize)
            slope[i,j] = np.degrees(np.arctan(np.sqrt(dzx**2 + dzy**2)))
            
    return(slope)
    
    
@nb.njit(parallel=True)
def STDS(DEM, w, cellsize):
    
    stdsGrid = np.nan*DEM
    slope = Slope_grd(DEM,w,cellsize)
    [nrows, ncols] = np.shape(DEM)
    nw = (w*2+1)**2

    
    for i in nb.prange(w,nrows-w):
        for j in range(w,ncols-w):
            d1 = np.arange(i-w,i+w)
            d2 = np.arange(j-w,j+w)
            d3 = slope[d1[0]:d1[-1],d2[0]:d2[-1]]
            nz = d3.shape[0] * d3.shape[1]
            rms = np.sqrt(np.nansum((d3-np.nanmean(d3))**2)/nz)
            stdsGrid[i,j] = rms
    return(stdsGrid)
            