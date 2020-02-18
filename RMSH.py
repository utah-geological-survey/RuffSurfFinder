# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:27:21 2019
This function is based on code developed by Matteo Berti. It was subsequently heavily modified and parallelized for efficiency. The method looks for the root-mean squared height within a moving window. It is an amplitude based method for calculating surface roughness. 

INPUTS:
    DEM - a square shaped digital elevation model imported into python
    w - window size over which the STDS will be calculated

    
OUTPUTS:
    RMSH Grid - the grid of calculated root mean squared height.
    
References:
    Shepard et al., 2001,The roughness of natural terrain: A planetary and remote sensing perspective, Journal of Geophysical Research: Planets, V 106, E12, P. 32777-32795
@author: matthewmorriss
"""
import numpy as np
import numba as nb

@nb.njit()
def detrend(w):
    Npts=w.shape[0]
    A=np.empty((Npts,2),dtype=w.dtype)
    for i in range(Npts):
        A[i,0]=1.*(i+1) / Npts
        A[i,1]=1.

    coef, resids, rank, s = np.linalg.lstsq(A, w.T)
    out=w.T- np.dot(A, coef)
    return out.T

@nb.njit()
def isnan(win):
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            if np.isnan(win[i,j]):
                return True
    return False


@nb.njit(parallel=True)
def RMSH(DEM, w):
#    import progress as progress

    [nrows, ncols] = np.shape(DEM)
    
    #create an empty array
    rms = DEM*np.nan
    
#    nw=(w*2)**2
#    x = np.arange(0,nw)
    
    [nrows, ncols] = np.shape(DEM)

    #create an empty array to store result
    rms = DEM*np.nan

    for i in nb.prange(w+1,nrows-w):
#        total  = nrows-w
#        progress.progress(i,total,'Doing long job')
        for j in range(w+1,ncols-w):
            win = DEM[i-w:i+w-1,j-w:j+w-1]

            if isnan(win):
                rms[i,j] = np.nan
            else:
                win = detrend(win)
                z = win.flatten()
                nz = z.size
                rootms = np.sqrt(1 / (nz - 1) * np.sum((z-np.mean(z))**2))
                rms[i,j] = rootms

        
    return(rms)