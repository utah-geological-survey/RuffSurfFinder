# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:02:43 2019


This function computes the 2D CWT of a dem using the Mexican Hat Wavelet. 

INPUTS:
    Patch - piece of a dem
    a- wavelet scale
    dx = grid spacing
    
OUTPUTS:
    C - filtered DEM
    frq - bandpass frequency of wavelet at scale a
    wave - wavelegnth (inverse of frequency)
    
Heavily modified from original code by Adam Booth @ PSU

@author: matthewmorriss
"""

def conv2_mexh2(dem,a,dx):
    import numpy as np
    from scipy import signal
    import time
    start = time.time()
    #Generated the Mexican hat wavelet kerne at wavelet scale a.  The kernal much be large enough for the wavelet to decay to ~0 at the edges. The Mexican hat is proportional to the second derivative of a gaussian
    [X,Y] = np.meshgrid(np.arange(-8*a,8*a),np.arange(-8*a,8*a))
    psi = (1/a)*(2 - (X/a)**2 - (Y/a)**2)*(np.exp(-((X/a)**2 + (Y/a)**2)/2))


#    C = (dx**2)*signal.convolve2d(dem, psi,'same')

    C = (dx**2)*signal.fftconvolve(dem,psi,'same')
    
#    dem[dem == -9999.0] = np.nan
#    dem[dem == -32767] =np.nan
    [nrows, ncols] = np.shape(dem)
    fringeEval = np.ceil(4*np.max(a))
        
    C[(np.arange(0,fringeEval)).astype(int),:] = np.NaN
    C[:,(np.arange(0,fringeEval)).astype(int)] = np.NaN
    C[np.arange((nrows-fringeEval),nrows).astype(int),:] = np.NaN
    C[:,(np.arange(ncols-fringeEval,ncols)).astype(int)] = np.NaN
    
    wave = 2*np.pi*dx*a/((5/2)**(1/2))
    frq = 1/wave
    end = time.time()
    print(end-start)
    return(C, frq, wave)