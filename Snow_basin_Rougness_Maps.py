# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:04:40 2019
This is a script written to test the efficacy of four different methods of roughness and landslide mapping

METHODS:
    1) Continuous Wavelet Transform     (CWT)
    2) Standard Deviation of Slope      (STDS)
    3) Root mean square height          (RMSH)
    4) Directional Cosine Eigenvector   (DCE)
    
    
The general setup will be a single function call to each roughness method, to create a surface roughness map of the Snowbasin Ski area. This is a region with several known landslides which have been mapped and monitored by the UGS over the last decade to decade-and-a-half. 

More information on these landslides can be viewed in either the technical report written by Matthew Morriss for UGS publication, or is described briefly, here: https://geology.utah.gov/map-pub/survey-notes/green-pond-landslide/.

DEFINITIONS:
    SBROI - Snow Basin Region of Interest
    
INPUTS:
   ##### IMPORTANTLY!!##### ALL INPUT FILES ARE EXPORTED FROM ARCGIS IN UTM COORDS
    
    1) sb_less_steep.tif -- This is a 2 m resolution DEM of the SBROI. 
    2) ls_patch_4.tif -- A small piece of a landslide in the SBROI
    3) no_ls_patch4.tif -- A small piece of non-landslide terrain in the SBROI.
    
OUTPUTS:
    Each cell that calls a roughness method will output a surface roughness map for the Snowbasin Region of Interst
    
REQUIRED PYTHON PACKAGES:
    time
    os
    numpy
    matplotlib
    rasterio
    osgeo
    pandas
    numba
    
CUSTOM FUNCTIONS:
    conv2_mexh_var
    conv2_mexh
    hillshade
    progress
    STDS
    RMSH
    smooth2
    DCE_par
    DCE_preprocess
    
    DEPENDENCIES:
        
@author: matthewmorriss
"""



import time
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio as rio
from osgeo import gdal

# Set your working directory to the folder containing all of the requisite codes
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\Surface_Roughness_code')

#import all necessary custom functions
from hillshade import hillshade
from conv2_mexh_var import conv2_mexh_var
from conv2_mexh2 import conv2_mexh2
from smooth2 import smooth2
from STDS import STDS
from RMSH import RMSH
from DC_eig_par import DC_eig_par
from DCE_preprocess import DCE_preprocess




lspatch = 'sb_less_steep.tif' # <------ INPUT FILE NAME <------

#manually set the dx or cell size, uncomment if you want otherwise it will be detected by Gdal
# cellsize = 0.5 #<-------
# dx = cellsize

# call gdal function to upload DEM file.
dem_path = os.path.join(os.getcwd(),lspatch)
DEM = gdal.Open(dem_path)
gt = DEM.GetGeoTransform()
dx = gt[1]
cellsize = dx
DEM = np.array(DEM.GetRasterBand(1).ReadAsArray()) #convert gdal obj into np array
del(gt)

#import meta data regarding raster, for eventual exporting of filtered DEM
import rasterio as rio
with rio.open(dem_path) as src:
    Meta = src.profile

#%% Visualize the imported DEM to confirm it appears as you expect
    # the hillshade produced will be rather dark.

hs_array = hillshade(DEM)

#Plot hillshade
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(hs_array,cmap='Greys', alpha = 0.9)
ax.plot()


#%%  TEST 1: CWT
# This test requires patches of both the landslide and non-landslide to compare the power spectra between the two pieces of the landscape.

# This methodology follows Booth et al., (2009).
#### IMPORTANTLY!!! #### EACH TEST PATCH MUST BE CLIPPED SQUARE!
#This code is heavily adapted from code available here: http://web.pdx.edu/~boothad/tools.html
# It was originally writen in Matlab, but I have now adapted it for Python.

#Import LANDSLIDE patch
dem_path = os.path.join(os.getcwd(),'ls_patch_4.tif')
demFLD = gdal.Open(dem_path)
demFLD = np.array(demFLD.GetRasterBand(1).ReadAsArray())

#Import NON-LANDSLIDE patch
dem_path = os.path.join(os.getcwd(),'no_ls_p4.tif')
demUNfld = gdal.Open(dem_path)
demUNfld = np.array(demUNfld.GetRasterBand(1).ReadAsArray()) #this removes


# Set scales for different wavelet sizes. These are convertable to wavelengths through the following equation:
# W = 2*np.pi*dx*s/(np.sqrt(5/2))
# where:
#       s = scales

scales = np.exp(np.linspace(0,4,20))
#The conv2_mexh_var convolves the dem patch with a wavelet of given scales and then outputs the spectral wavelength.
[Vcwt_fld, frq, wave] = conv2_mexh_var(demFLD, scales, dx)
[Vcwt_unfld, _, _] = conv2_mexh_var(demUNfld, scales, dx)


Vcwt_fld = np.transpose(Vcwt_fld)
Vcwt_unfld = np.transpose(Vcwt_unfld)
Vcwt_norm = Vcwt_fld/Vcwt_unfld
#%% PLOT CWT RESULTS

from matplotlib.ticker import FormatStrFormatter
def spec2m(x):
    return(1/x)
    
def m2spec(x):
    return(1/x)

#Plot 1 wavelet power spectra
fig1 = plt.figure()
ax1=fig1.add_subplot(1,1,1)
ax1.loglog(frq, Vcwt_fld,'s', label = 'Landslide terrane')
ax1.loglog(frq,Vcwt_unfld,'v', label = 'Non Landslide')
ax1.legend()
#ax1.set_xticks( np.arange(10**-1,1,10**-1), minor = False)
ax1.set_xticks([1.0, 0.1, 0.01, 0.001])
ax1.grid(axis = 'x')
ax1.set_xlim(1e-3,1)
secax = ax1.secondary_xaxis('top',functions = (spec2m,m2spec))
secax.set_xlabel('Distance (m)')
secax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax1.set_xlabel('Frequency (m$^{-1}$)')
ax1.set_ylabel('CWT Power')
ax1.grid(which = 'both')
#ax1.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax1.set_title('Wavelet Power Spectra')



# Plot 2 normalized wavelet power spectra. This plot compares the power of the landsldie patch with the non-landslide path.
fig2 = plt.figure()
ax2=fig2.add_subplot(1,1,1)
ax2.semilogx(frq, Vcwt_norm,'ko-')
ax2.set_xticks([1.0, 0.1, 0.01, 0.001])
#ax2.set_xticks( np.arange(10**-1,10e-3,10**-1,), minor = False)
ax2.grid(axis = 'x')
ax2.grid(which = 'both')
ax2.set_xlim(1e-3,1.0)
secax = ax2.secondary_xaxis('top',functions = (spec2m,m2spec))
secax.set_xlabel('Distance (m)')
ax2.set_ylabel('Normalized Power')
ax2.set_xlabel('Frequency (m$^{-1}$)')
#ax2.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax2.set_title('Normalized Wavelet Spectra')

"""If you follow the logic of the full width at half maximum wavelength logic laid out by Booth et al., 2009, the result would be the wavelengths of 14-46 m are ideal for detecting the spectral character of landslides. However, I have tested these wavelengths and found that they do not provide the best resolution, so I opted to use a narrower wavelength spread for this example. This is likely because I opted to use a longer wavelength example of a landslide, so the smaller scale hummocks of the earthflows on the west and southwestern side of the SBROI are missed

Below, I use 7-13 m in a power sum to make a roughness map """
#%% First clean up workspace by deleting unnecesary variables
del(ax1,ax2,Vcwt_fld, Vcwt_norm, Vcwt_unfld, fig1, fig2)
#%% CWT WITH A 7-13 M POWERSUM
###################################################
###################################################
##### THIS CORRESPONDS TO 7 - 13 M ROUGHNESS ####
###################################################
###################################################   

#Conv2_mexh2 convolves a mexican hat wavelet of a chosen scale with the full DEM
#There is not wait bar for this step; however, you will be provided a print out in seconds for how long each convolution has taken.

# BELOW ARE THE EQUATIONS TO GO FROM WAVELET SCALE TO SPATIAL WAVELENGTH
# W = 2*np.pi*dx*s/(np.sqrt(5/2))
#S = (w * np.sqrt(5/2)/(2*np.pi*dx))
[C2,_,_] = conv2_mexh2(DEM, 7,dx)
[C3,_,_] = conv2_mexh2(DEM, 5,dx)
[C4,_,_] = conv2_mexh2(DEM, 4,dx)



# Square and sum wavelet coefficients in quadrature

Vcwtsum = C2**2 + C3**2 + C4**2 
del(C2,C3,C4)

#PLOT FIGURE 1, THE RAW POWERSUM ROUGHNESS MAP
#create a masked array to not include the NaNs in the color ramp
masked_array = np.ma.array(Vcwtsum, mask = np.isnan(Vcwtsum))

fig1, ax1= plt.subplots() 
current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)

ax1 = plt.imshow(np.log(Vcwtsum),cmap = current_cmap)
ax1.set_array(masked_array)
del(masked_array)

#PLOT FIGURE 2, THE SMOOTHED POWERSUM, MORE COMPARABLE TO DFT RESULTS

radius = 25
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)
masked_array = np.ma.array(Vcwt_smooth, mask = np.isnan(Vcwt_smooth))

fig2, ax2 = plt.subplots()
current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)

ax2 = plt.imshow(np.log(Vcwt_smooth),cmap = current_cmap)
ax2.set_array(masked_array)
ax2.set_title('CWT Power Sum (smoothed)')


### EXPORT SMOOTHED CWT IMAGE #####
fName = 'CWT'  + '_rghness.tif'
lsMap = np.float32(np.log(Vcwt_smooth))
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
""""I recommend that you import the exported tif into a GIS program which is capable of displaying raster data for ultimate visualization. Python is not the best at imaging"""
    

#%% CWT for 14-46 M
"""UNCOMMENT CODE BELOW TO SEE THE WIDER SPECTRUM RESULTS"""

# ###################################################
# ###################################################
# ##### THIS CORRESPONDS TO 14 - 46 M ROUGHNESS ####
# ###################################################
# ###################################################
# [C1,_,_] = conv2_mexh2(DEM, 7,dx)
# [C3,_,_] = conv2_mexh2(DEM, 11,dx)
# [C5,_,_] = conv2_mexh2(DEM, 15,dx)
# [C7,_,_] = conv2_mexh2(DEM, 19,dx)
# [C9,_,_] = conv2_mexh2(DEM, 23,dx)

# # Square and sum wavelet coefficients in quadrature
# Vcwtsum = C1**2 + C3**2  + C5**2   + C7**2 + C9**2 
# del(C1,C3,C5,C7,C9)           

# #PLOT FIGURE 1, RAW POWERSUM
# masked_array = np.ma.array(Vcwtsum, mask = np.isnan(Vcwtsum))

# fig1, ax1= plt.subplots() 
# current_cmap = matplotlib.cm.RdYlBu
# current_cmap.set_bad('black',1.)

# ax1 = plt.imshow(np.log(Vcwtsum),cmap = current_cmap)
# ax1.set_array(masked_array)
# del(masked_array)

# #PLOT FIGURE 2, SMOOTHED POWERSUM
# radius = 25
# [Vcwt_smooth,_] = smooth2(Vcwtsum,radius)
# masked_array = np.ma.array(Vcwt_smooth, mask = np.isnan(Vcwt_smooth))

# fig2, ax2 = plt.subplots()
# current_cmap = matplotlib.cm.RdYlBu
# current_cmap.set_bad('black',1.)

# ax2 = plt.imshow(np.log(Vcwt_smooth),cmap = current_cmap)
# ax2.set_array(masked_array)
# ax2.set_title('CWT Power Sum (smoothed)')

#%% Cleanup workspace
del(Vcwtsum, Vcwt_smooth,demFLD, demUNfld,masked_array,wave, ax1,ax2,fig1,fig2,frq,radius)

#%% TEST 2, STANDARD DEVIATION OF SLOPE
""""Below is the test of the standard deviation of slope, which requires a window size and then calls the STDS function. This method is developed out of the testing of Berti et al., 2013, which cites Frankel and Dolan 2007.""" 

# Set moving window size
w = 20 

#calculate STDS GRID
stdGrd = STDS(DEM,w,cellsize)

#Export grid as .tif
fName = 'STDS_w_' + str(w) + '.tif'
stdGrd = np.float32(stdGrd)


#Plot standard deviation of slope grid
stdGrd[stdGrd == 0] = np.nan  
masked_array = np.ma.array(stdGrd, mask = np.isnan(stdGrd))
fig1, ax1 = plt.subplots()

current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)
ax1 = plt.imshow(stdGrd,cmap = current_cmap)
ax1.set_array(masked_array)
del(masked_array)

#Export STDS Grid
with rio.open(fName,'w',**Meta) as dst:
    dst.write(stdGrd,1)

#%% Cleanup workspace
    
del(stdGrd, ax1, fig1, masked_array)
#%% TEST 3, ROOT MEAN SQUARED HEIGHT
"""This is a test of the root meant squared height method of measuring surface roughness. This method is developed from code written by Berti et al., 2013 which built on the methods described in Shepard et al,. 2001. As with other methods, the resulting grid is best visualized in a GIS software that can visualize different raster datasets."""

#set moving window size
w = 20

#calculate RMSH GRID
rmshGrd = RMSH(DEM,w)


#Plot root mean squared height grid
rmshGrd[rmshGrd == 0.0034124573751434595] = np.nan  
fig1, ax1 = plt.subplots()

current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)
ax1 = plt.imshow(np.log(rmshGrd),cmap = current_cmap)



#export grid as .tif
fName = 'RMSH_w_' + str(w) + '.tif'
rmshGrd = np.float32(rmshGrd)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(rmshGrd,1)

#%% Cleanup  workspace
  
del(fig1,ax1,rmshGrd)
#%% TEST 4 DIRECTIONAL COSINE EIGENVECTOR

#set window size
w = 20

# calculate DCE GRID
[cx,cy,cz] = DCE_preprocess(DEM,cellsize,w)
eps = np.finfo(float).eps
eps = np.float32(eps)
eigGrd = DC_eig_par(DEM, w,cx,cy,cz,eps)


#Plot DCE result
eigGrd[eigGrd == 0.0034124573751434595] = np.nan  
fig1, ax1 = plt.subplots()

current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)
ax1 = plt.imshow(np.log(eigGrd),cmap = current_cmap)




#Export DCE Grid as a .tif
fName = 'DCE_w_' + str(w) + '.tif'
eigGrd = np.float32(eigGrd)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(eigGrd,1)
        
   