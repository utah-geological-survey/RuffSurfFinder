# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:20:28 2019
The actual work of creating the wavelet and conducting the 2d convolution

@author: matthewmorriss
"""

def conv2_mexh(patch,a,dx):
    import numpy as np
    from scipy import signal
    #Generated the Mexican hat wavelet kerne at wavelet scale a.  The kernal much be large enough for the wavelet to decay to ~0 at the edges. The Mexican hat is proportional to the second derivative of a gaussian
    [X,Y] = np.meshgrid(np.arange(-8*a,8*a),np.arange(-8*a,8*a))
    psi = (1/a)*(2 - (X/a)**2 - (Y/a)**2)*(np.exp(-((X/a)**2 + (Y/a)**2)/2))

    # TO PLOT PSI UNCOMMENT
#    from matplotlib import cm
#    ax = plt.axes(projection = "3d")
#    ax.plot_surface(X, Y, psi,cmap=cm.coolwarm)
#   C = 
    # TRYING TO FIGURE OUT THE MOST EFFICIENT 2D CONVOLUTION
#    start = time.time()    
#    C = (dx**2)*signal.convolve2d(patch, psi,'same')
#    end = time.time()
#    print(end-start)
    
    
#    N.B. to avoid issues with FFT convolve there cannot be any NaNs in your array otherwise the function will return only NaNs
    C = (dx**2)*signal.fftconvolve(patch,psi,'same')
# 
    
    return(C)