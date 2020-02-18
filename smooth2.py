# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:25:26 2019
This function will complete the simple smoothing used by Adam Booth in his 2009 paper. The original code can be accessed at Adam Booth's website http://web.pdx.edu/~boothad/tools.html. This function was ported from Matlab.

INPUTS:
    M = an array to be smoothed
    radius = smoothing radius
    
OUTPUTS:
    Msmooth - Smoothed grid
    Mres - residuals between original and smoothed grid
    
References:
   Booth et al., 2009, Automated landslide mapping using spectral analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland Hills, Oregon, Geomorphology, v. 109, no. 3-4, p. 132-147.

@author: matthewmorriss
"""

def smooth2(M, radius):
    import numpy as np
    from scipy import signal
    
    kernel = np.zeros((2*radius, 2*radius))
    [X,Y] = np.meshgrid(np.arange(-radius,radius), np.arange(-radius, radius))
    kernel[(X**2 + Y**2)**0.5 <= radius] = 1
    
    nans = np.where(np.isnan(M))
    M[nans] = 0
    # Filter M with the kernel using the 2d convolution
    Msmooth = signal.fftconvolve(M, kernel,'same')
#    Msmooth = signal.convolve2d(M, kernel,'same')
    
#    divide by the sum of all nodes in kernel to get an average at each node
    Msmooth = Msmooth/np.sum(kernel)
    
    #subtract to get residuals
    nans = np.where(M == 0)
    Msmooth[nans] = np.nan
    
    Mres = M - Msmooth
    
    return(Msmooth, Mres)