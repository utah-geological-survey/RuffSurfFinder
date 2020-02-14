# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:39:14 2019
Computes the wavelet spectrum (wavelet variance v. scale) of path over specified scales using the Mexican hat wavelet.  Note that the edge effects increase with scale, so Vcwt at larger scales become's increasingly biased towards nodes in the center of the patch.

Inputs: 
    Patch = patch of digital elevation model, typically a piece of landslide or non         landslide
    scales = wavelet scales over which the Vcwt is computer*
    dx = grid spacing

Outputs:
    Vcwt = vector of wavelet variance computer at each scale
    frq = vector of frequences
    wave = vector of wavelengths
    

* Set scales for different wavelet sizes. These are convertable to wavelengths through the following equation:
W = 2*np.pi*dx*s/(np.sqrt(5/2))
where:
      s = scales
      
This code is largely based on the methods described by Adam Booth in his 2009 code released with the Booth et al., (2009) paper. See http://web.pdx.edu/~boothad/tools.html for more details. This original code was written in Matlab.

@author: matthewmorriss written 9/17/19
"""

def conv2_mexh_var(patch, scales, dx):
    
    import progress
    import numpy as np
    from conv2_mexh import conv2_mexh

    patch[patch == -9999.0] = 0
    patch[patch == -32767] = 0
    patch[np.isnan(patch)] = 0
    [nrows, ncols] = np.shape(patch)
    nodes = nrows * ncols
    
    #Normalize patch to have unit variance
    patch = patch/np.nanstd(patch)
    
    #initialize the output vectors:
    Vcwt = np.zeros((1,np.size(scales)))
    
    #Determine extent of edge effecst at largest wavelt scale sampled. NaN values will be assigned to the fringe of each C grid in the loop so that the same number of nodes are used at each scale for determining Vcwt:
    fringeEval = np.ceil(4*np.max(scales))
    
    
    #start counter
    k = 0
    for a in scales:
        progress.progress(a,np.max(scales),'Doing long job')
        
        #update counter
       
        
        #Compute the 2D CWT by calling Conv2_mexh function (below)
        C = conv2_mexh(patch,a,dx)
        
        # Mask edge effects with naN (no Data)
        C[(np.arange(0,fringeEval)).astype(int),:] = np.NaN
        C[:,(np.arange(0,fringeEval)).astype(int)] = np.NaN
        C[np.arange((nrows-fringeEval),nrows).astype(int),:] = np.NaN
        C[:,(np.arange(ncols-fringeEval,ncols)).astype(int)] = np.NaN
    
        #find NaNs and replace with 0        
        ind = np.argwhere(np.isnan(C))
        C[np.isnan(C)] = 0
        
        #now calculate the wavelet variance at current scale, using number of real-valued nodes
        Vcwt[0,k] = 1/(2*(nodes - ind.shape[0]))*np.sum(np.sum(C**2,1),0)
        
        #frequency and wavelegth vectors
        wave = 2*np.pi*dx*scales/(5/2)**(1/2)
        frq = 1/wave
        k = k +1
    
    return(Vcwt,frq,wave)

