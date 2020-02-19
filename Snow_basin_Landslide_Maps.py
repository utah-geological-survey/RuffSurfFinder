# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:04:40 2019

This is a script written to make surface roughness maps and test the efficacy of four different methods of four measuring surface roughness. These methods will then be tested using the Receiver Operating Characteristic (ROC) Curve to compare these results to an a priori map of landsliding.  

METHODS:
    1) Continuous Wavelet Transform     (CWT)
    2) Standard Deviation of Slope      (STDS)
    3) Root mean square height          (RMSH)
    4) Directional Cosine Eigenvector   (DCE)
    
    
The general setup will be a series of experiments using different window sizes with each technique to create a surface roughness map of the Snowbasin Ski area. This is a region with several known landslides which have been mapped and monitored by the UGS over the last decade to decade-and-a-half. Then within the executed loop, an ROC curve will be constructed by calling the ROC_plot function. This loops through an array of roughness thresholds.

More information on these landslides can be viewed in either the technical report written by Matthew Morriss for UGS publication, or is described briefly, here: https://geology.utah.gov/map-pub/survey-notes/green-pond-landslide/.

DEFINITIONS:
    SBROI - Snow Basin Region of Interest
    
INPUTS:
   ##### IMPORTANTLY!!##### ALL INPUT FILES ARE EXPORTED FROM ARCGIS IN UTM COORDS
    
    1) sb_less_steep.tif -- This is a 2 m resolution DEM of the SBROI. 
    2) ls_patch_4.tif -- A small piece of a landslide in the SBROI
    3) no_ls_patch4.tif -- A small piece of non-landslide terrain in the SBROI.
    4) ls_boolean_map_match.tif -- a map of 0s and 1s, with the same dimensions as the original DEM (sb_less_steep.tif) 
    
OUTPUTS:
    ROC Curve - for each method a figure that has a curve plotted for each window size tested
    Landslide Maps - maps derived from the ROC curve threshold of roughness. 
    
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
    ROC_plot
    
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
import pandas as pd

# Set your working directory to the folder containing all of the requisite codes
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\RuffSurfFinder')

#import all necessary custom functions
from hillshade import hillshade
from conv2_mexh_var import conv2_mexh_var
from conv2_mexh2 import conv2_mexh2
from smooth2 import smooth2
from STDS import STDS
from RMSH import RMSH
from DC_eig_par import DC_eig_par
from DCE_preprocess import DCE_preprocess
from ROC_plot import ROC_plot

lspatch = 'sb_less_steep.tif' # <------ INPUT FILE NAME <------
boolPatch = 'ls_boolean_map_match.tif' #<------Input boolean file name 

# Import DEM of SBROI
dem_path = os.path.join(os.getcwd(),lspatch)
DEM = gdal.Open(dem_path)
gt = DEM.GetGeoTransform()
dx = gt[1]
cellsize = dx
DEM = np.array(DEM.GetRasterBand(1).ReadAsArray())

#import meta data regarding raster, for eventual exporting of filtered DEM
import rasterio as rio
with rio.open(dem_path) as src:
    Meta = src.profile


#resample boolean map to match the dimensions of the roughness map (key)
from rasterio.enums import Resampling
with rio.open(boolPatch) as dataset:
    data = dataset.read(
            out_shape = (np.shape(DEM)[0], \
                         np.shape(DEM)[1]),resampling = Resampling.bilinear
    )

lsRaster = data[0,:,:]
lsRaster = np.float64(lsRaster)
lsRaster[np.isnan(lsRaster)] = 0 #should now be a map of 1s and 0s
#%% Plot the overlaid boolean array over the hillshade of the DEM
# this is an important step to be sure your boolean array is of the correct dimensions.

hs_array = hillshade(DEM)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(hs_array,cmap='Greys')
ax.imshow(lsRaster, alpha = 0.3)
ax.plot()

#%%  TEST 1: CWT
# This test requires patches of both the landslide and non-landslide to compare the power spectra between the two pieces of the landscape.

# This methodology follows Booth et al., (2009).
#### IMPORTANTLY!!! #### EACH TEST PATCH MUST BE CLIPPED SQUARE or in a rectangle!
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

"""If you follow the logic of the full width at half maximum wavelength logic laid out by Booth et al., 2009, the result would be the wavelengths of 12-50 m are ideal for detecting the spectral character of landslides. I will also test a series of other wavelength bands: 18 - 25 and 8 - 14 m bands as well. Several sections lower there is a place for running a combined ROC plot. """

#%% TEST 1A FOR CWT
###################################################
###################################################
##### THIS CORRESPONDS TO 12 - 50 M ROUGHNESS ####
###################################################
###################################################
# W = 2*np.pi*dx*s/(np.sqrt(5/2))
#S = (w * np.sqrt(5/2)/(2*np.pi*dx))

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

[C1,_,_] = conv2_mexh2(DEM, 7,dx)
[C3,_,_] = conv2_mexh2(DEM, 11,dx)
[C5,_,_] = conv2_mexh2(DEM, 15,dx)
[C7,_,_] = conv2_mexh2(DEM, 19,dx)
[C9,_,_] = conv2_mexh2(DEM, 23,dx)

# Square and sum wavelet coefficients in quadrature
Vcwtsum = C1**2 + C3**2  + C5**2   + C7**2 + C9**2 
del(C1,C3,C5,C7,C9)           

#PLOT FIGURE 1, RAW POWERSUM
masked_array = np.ma.array(Vcwtsum, mask = np.isnan(Vcwtsum))

fig1, ax1= plt.subplots() 
current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)

ax1 = plt.imshow(np.log(Vcwtsum),cmap = current_cmap)
ax1.set_array(masked_array)
del(masked_array)

#PLOT FIGURE 2, SMOOTHED POWERSUM
radius = 25
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)
masked_array = np.ma.array(Vcwt_smooth, mask = np.isnan(Vcwt_smooth))

fig2, ax2 = plt.subplots()
current_cmap = matplotlib.cm.RdYlBu
current_cmap.set_bad('black',1.)

ax2 = plt.imshow(np.log(Vcwt_smooth),cmap = current_cmap)
ax2.set_array(masked_array)




#### ROC CURVE PLOT BELOW ###
lowT = 2        #LOW THRESHOLD
highT = 11      #HIGH THRESHOLD
numDiv = 40     #NUM OF DIVISIONS
lsMap = np.float32(np.log(Vcwt_smooth))
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve

#SAVE RESULTING LANDSLIDE MAP
fName = 'CWT'  + '_lsmap_14_t_8.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)

#%% clean up workspace
del(Vcwtsum, Vcwt_smooth,masked_array, ax1,ax2,fig1,fig2,frq,radius)

#%% TEST 1B OF CWT
###################################################
###################################################
##### THIS CORRESPONDS TO 14 - 8 M ROUGHNESS ####
###################################################
###################################################   
[C2,_,_] = conv2_mexh2(DEM, 7,dx)
[C3,_,_] = conv2_mexh2(DEM, 5,dx)
[C4,_,_] = conv2_mexh2(DEM, 4,dx)



# Square and sum wavelet coefficients in quadrature

Vcwtsum = C2**2 + C3**2 + C4**2 
del(C2,C3,C4)

fig1, ax1= plt.subplots()
ax1.imshow(np.log(Vcwtsum)) 

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)

fig2, ax2 = plt.subplots()
ax2.imshow(np.log(Vcwt_smooth))

    

#### ROC CURVE PLOT BELOW ###
lowT = -2       #LOW THRESHOLD
highT = 5       #HIGH THRESHOLD
numDiv = 40     #NUM OF DIVISIONS
lsMap = np.float32(np.log(Vcwt_smooth))
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)


auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve

# Export resulting landslide map.
fName = 'CWT'  + '_lsmap_14_t_8.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)

#%% clean up workspace
del(Vcwtsum, Vcwt_smooth,masked_array, ax1,ax2,fig1,fig2,frq,radius)

#%% TEST 1C OF CWT
###################################################
###################################################
##### THIS CORRESPONDS TO 18 - 25 M ROUGHNESS ####
###################################################
###################################################   
[C2,_,_] = conv2_mexh2(DEM, 9,dx)
[C3,_,_] = conv2_mexh2(DEM, 10,dx)
[C4,_,_] = conv2_mexh2(DEM, 11,dx)
[C5,_,_] = conv2_mexh2(DEM, 12.5,dx)



# Square and sum wavelet coefficients in quadrature
Vcwtsum = C2**2 + C3**2 + C4**2 + C5**2

del(C2,C3,C4,C5)

#plot raw wavelet sums
fig1, ax1= plt.subplots()
plt.imshow(np.log(Vcwtsum))
ax1.set_title('CWT Spectral Power Sum')

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)

fig2, ax2 = plt.subplots()
ax2.set_title('CWT Power Sum (smoothed)')
ax2.imshow(np.log(Vcwt_smooth))


#### ROC CURVE PLOT BELOW ###
lowT = 0        #LOW THRESHOLD
highT = 8.1     #HIGH THRESHOLD
numDiv = 40     #NUMBER OF DIVISIONS
lsMap = np.float32(np.log(Vcwt_smooth))
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve


#EXPORT LANDSLIDE MAP
fName = 'CWT'  + '_lsmap_18_t_25.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)

#%% clean up workspace
del(Vcwtsum, Vcwt_smooth, ax1,ax2,fig1,fig2,radius)


#%% Plot all ROCS for CWT
winds = ["12-50 m", "14 - 8 m","18 - 25 m"]
zz = np.vstack([np.asarray(aucL), np.asarray(tMaxL),winds])
zz = zz.T


DF = pd.DataFrame(zz)
DF.columns = ['AUC','Rc','window size']
DF.to_excel('windowSize_CWT.xlsx',header = True,index = False)


# Plot resulting ROC curves
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#colours=['r','g','b','k']
for i in np.arange(0,len(tprL)):
    lab = 'Window = ' + winds[i] + ' ' + ' AUC = ' + str(round(aucL[i],2))
    ax.plot(fprL[i],tprL[i],label = lab)
    ax.legend()
fig.suptitle('STDS windowed experiments, ROC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

figName = 'CWT_ROC_curve.pdf'
plt.savefig(figName)

#%% clean up workspace
del(fig, ax, zz, DF, winds, Vcwt_fld, Vcwt_norm, Vcwt_unfld,aucL, fprL, tprL, tMaxL, auc, tpr, lowT, highT, numDiv)
o
#%% TEST 2 STANDARD DEVIATION OF SLOPE
""""Below is the test of the standard deviation of slope, which requires a window size and then calls the STDS function. This method is developed out of the testing of Berti et al., 2013, which cites Frankel and Dolan 2007.

This function will be run over a series of window sizes (5 -35). The experiments with different window sizes are conducted within a For loop for simplicity and then the ROC plot is generated.
""" 


fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 35       #MAXIMUM WINDOW SIZE
minW = 5        #MINIMUM WINDOW SIZE

#RUN FOR LOOP FOR STANDARD DEVIATION OF SLOPE
for i in np.arange(minW,maxW,5):
    
    #calculate STDS grid
    stdGrd = STDS(DEM,i,cellsize)
     
    # PLOT ROC CURVE
    lowT = 0        #LOW THRESHOLD
    highT = 20      #HIGH THRESHOLD
    numDiv = 40     #NUMBER OF DIVISIONS
    [fpr,tpr, lsMap, tmax] = ROC_plot(stdGrd,lsRaster,lowT,highT,numDiv)
    
    #SAVE RESULTS FROM ROC FOR LATER
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    plt.plot(fpr,tpr)
    
    #SAVE RESULTING LANDSLIDE MAP
    fName = 'STDS_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

#outside of loop compile all results from ROC curve
zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T

#compile results into a table and export to excel
DF = pd.DataFrame(zz)
DF.columns = ['Window', 'AUC','Rc']
DF.to_excel('windowSize_STDS.xlsx',header = True,index = False)

# PLOT ALL ROC CURVES IN SAME PLOT
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#colours=['r','g','b','k']
for i in np.arange(0,len(tprL)):
    lab = 'Window = ' + str(wL[i]) + 'm' + ' AUC = ' + str(round(aucL[i],2))
    ax.plot(fprL[i],tprL[i],label = lab)
    ax.legend()
fig.suptitle('STDS windowed experiments, ROC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

#SAVE PLOT
figName = 'STDS_ROC_Curves.pdf'
plt.savefig(figName)


#%% clean up workspace
del(fig, fpr, fprL, highT, lowT, numDiv, i, auc, aucL, radius, zz, DF, ax,stdGrd, tMaxL, tmax, tpr, tprL)
#%% TEST 3 ROOT MEAN SQUARED HEIGHT
"""This is a test of the root meant squared height method of measuring surface roughness. This method is developed from code written by Berti et al., 2013 which built on the methods described in Shepard et al,. 2001. This code will generate a landslide map by comparing the surface roughness map to the roughness of the a priori mapped landslides.

The RMSH will be calculated on windows between 5 and 35 at increments of 5.

This will take about ~1 hour  to run on a fast machine with ~8 cores."""

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 35       #MAXIMUM WINDOW SIZE
minW = 5        #MINIMUM WINDOW SIZE

#RUN FOR LOOP FOR ROOT MEAN SQUARED HEIGHT
for i in np.arange(minW,maxW,5):
    
    # RUN RMSH
    stdGrd = RMSH(DEM,i)
    
    #PLOT ROC CURVE
    lowT = 0        #LOW THRESHOLD
    highT = 1       #HIGH THRESHOLD
    numDiv = 40     #NUMBER OF DIVISIONS
    [fpr,tpr, lsMap, tmax] = ROC_plot(stdGrd,lsRaster,lowT,highT,numDiv)
    
    # SAVE ROC RESULTS
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    #PLOT RESULTS
    plt.plot(fpr,tpr)
    
    #SAVE LANDSLIDE MAP FROM THIS MOVING WINDOW SIZE
    fName = 'RMSH_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

#COALATE ALL OF THE RESULTS FROM ROC
zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T

DF = pd.DataFrame(zz)
DF.columns = ['Window', 'AUC','Rc']
DF.to_excel('windowSize_RMSH.xlsx',header = True,index = False)

# Plot resulting ROC curves
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#colours=['r','g','b','k']
for i in np.arange(0,len(tprL)):
    lab = 'Window = ' + str(wL[i]) + 'm' + ' AUC = ' + str(round(aucL[i],2))
    ax.plot(fprL[i],tprL[i],label = lab)
    ax.legend()
fig.suptitle('RMSH windowed experiments, ROC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

#Save compiled plots
figName = 'RMHS_ROC_Curves.pdf'
plt.savefig(figName)

#%% clean up workspace
del(fig, fpr, fprL, highT, lowT, numDiv, i, auc, aucL, zz, DF, ax,stdGrd, tMaxL, tmax, tpr, tprL)


#%% TEST 4 DIRECTIONAL COSINE EIGENVECTOR
""""This is a test of the directional cosine eigen vector method of measuring surface roughness. This method was first described in McKean and Roering (2004). The resulting grids are best visualized in ArcGIS.

The DCE will be calculated across windows 10 to 45 in increments of 5."""

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 45       #maximum window size
minW = 10       #minimum window size
inc = 5         #increment size

for i in range(minW,maxW,inc):

    #preprocess DEM to calculate directional cosine grids
    [cx,cy,cz] = DCE_preprocess(DEM,cellsize,i)
    
    #calculate eps, machine relative error.
    eps = np.finfo(float).eps
    eps = np.float32(eps)
    
    #CALCULATE DIRECTIONAL COSINE EIGENVECTOR
    eigGrd = DC_eig_par(DEM, i,cx,cy,cz,eps)
    
    # PLOT ROC CURVE
    lowT = 0        #MAXIMUM THRESHOLD
    highT = 1       #MINIMUM THRESHOLD
    numDiv = 70     #NUMBER OF DIVISIONS
    [fpr,tpr, lsMap, tmax] = ROC_plot(eigGrd,lsRaster,lowT,highT,numDiv)
    
    #SAVE RESULTS FROM ROC
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    #PLOT ROC
    plt.plot(fpr,tpr)
    
    #EXPORT LANDSLIDE MAP FOR THIS ROC DERIVED THRESHOLD
    fName = 'DCE_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

#COALATE ALL ROC RESULTS INTO A SINGLE TABLE
zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T

#EXPORT RESULTS FROM ROC TO EXCEL FILE
DF = pd.DataFrame(zz)
DF.columns = ['Window', 'AUC','Rc']
DF.to_excel('windowSize_DCE.xlsx',header = True,index = False)

# Plot resulting ROC curves
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#colours=['r','g','b','k']
for i in np.arange(0,len(tprL)):
    lab = 'Window = ' + str(wL[i]) + 'm' + ' AUC = ' + str(round(aucL[i],2))
    ax.plot(fprL[i],tprL[i],label = lab)
    ax.legend()
fig.suptitle('DCE windowed experiments, ROC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

#SAVE RESULTING COMPILED ROC CURVE
figName = 'DCE_ROC_Curves.pdf'
plt.savefig(figName)