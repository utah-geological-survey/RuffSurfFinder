# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:04:40 2019

This is a script written to make surface roughness maps and test the efficacy of four different methods of for measuring surface roughness 

METHODS:
    1) Continuous Wavelet Transform     (CWT)
    2) Standard Deviation of Slope      (STDS)
    3) Root mean square height          (RMSH)
    4) Directional Cosine Eigenvector   (DCE)
    
    
The general setup will be a single function call to each roughness method, to create a surface roughness map of the Snowbasin Ski area. This is a region with several known landslides which have been mapped and monitored by the UGS over the last decade to decade-and-a-half. 

More information on these landslides can be viewed in either the technical report written by Matthew Morriss for UGS publication, or is described briefly, here: https://geology.utah.gov/map-pub/survey-notes/green-pond-landslide/.

DEFINITIONS:
    SBROI - Snow Basin Region of Interest
    
    
    
    
This is a script written to test the efficacy of four different methods of roughness and landslide mapping

METHODS:
    1) continuous wavelet transform
    2) Standard deviation of slope
    3) Root mean square height
    4) Directional Cosine Eigenvector
    
    
The general setup will be a loop for each technique, calculating the roughness and then moving onto the ROC curve plot

@author: matthewmorriss
"""




import time
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio as rio
from osgeo import gdal

lspatch = 'sb_less_steep.tif' # <------ INPUT FILE NAME <------
boolPatch = 'ls_boolean_map_match.tif' #<------Input boolean file name 

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt')
dem_path = os.path.join(os.getcwd(),lspatch)
DEM = gdal.Open(dem_path)
DEM = np.array(DEM.GetRasterBand(1).ReadAsArray())

import rasterio as rio
with rio.open(dem_path) as src:
    Meta = src.profile

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\Wavelets')
from conv2_mexh_var import conv2_mexh_var
from conv2_mexh import conv2_mexh
cellsize = 0.5

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt')
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


os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
#%%



#%% Visualize
from hillshade import hillshade
hs_array = hillshade(DEM)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(hs_array,cmap='Greys')
ax.imshow(lsRaster, alpha = 0.3)
ax.plot()
#%% ####### TEST OF STANDARD DEVIATION OF SLOPE ######
# window sizes = 5, 10, 15, 20
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
from ROC_plot import ROC_plot
from STDS import STDS
import progress as progress

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_roughness_map_sb')

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 35
minW = 5
for i in np.arange(minW,maxW,5):
    stdGrd = STDS(DEM,i,cellsize)
    fName = 'STDS_w' + str(i) + '.tif'
    stdGrd = np.float32(stdGrd)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(stdGrd,1)
        
    lowT = 0
    highT = 20
    numDiv = 40
    [fpr,tpr, lsMap, tmax] = ROC_plot(stdGrd,lsRaster,lowT,highT,numDiv)
    
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    plt.plot(fpr,tpr)
    
    fName = 'STDS_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T
import pandas as pd

DF = pd.DataFrame(zz)
DF.columns = ['Window', 'AUC','Rc']
DF.to_excel('windowSize_STDS.xlsx',header = True,index = False)

# Plot resulting ROC curves
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

figName = 'STDS_ROC_Curves.pdf'
plt.savefig(figName)

#%% ####### TEST OF RMSH roughness ######
# window sizes = 5, 10, 15, 20
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
from ROC_plot import ROC_plot
from RMSH_det_par_SO import RMSH_det_par_SO
import progress as progress

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_roughness_map_sb')

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 35
minW = 5
for i in np.arange(minW,maxW,5):
    stdGrd = RMSH_det_par_SO(DEM,i)
    fName = 'RMSH_w' + str(i) + '.tif'
    stdGrd = np.float32(stdGrd)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(stdGrd,1)
        
    lowT = 0 
    highT = 1
    numDiv = 40
    [fpr,tpr, lsMap, tmax] = ROC_plot(stdGrd,lsRaster,lowT,highT,numDiv)
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    plt.plot(fpr,tpr)
    
    fName = 'RMSH_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T
import pandas as pd

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
fig.suptitle('STDS windowed experiments, ROC')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

figName = 'RMHS_ROC_Curves.pdf'
plt.savefig(figName)

#%%
#%% USE windowed FFT to look at power of 
w = 47 # width of the window, must be odd
dx = 0.5 # cell size


os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
from fft_mean_spec import fft_mean_spec
normalize = 0
plots = 1


start = time.time()
### RUN FFT MEAN SPEC ON LS PATCH ###
[Vdftave_fld, Vdftvec_fld, fvec, freqmat] = fft_mean_spec(demFLD, 47, 0.5, normalize,plots)

### RUN FFT MEAN SPEC ON NO LS PATCH ###
[Vdftave_unfld, Vdftvec_unfld, fvec, freqmat] = fft_mean_spec(demUNfld, 47, 0.5, normalize,plots)

from fft_normPower import fft_normPower
#calculate normalized fourier
[Vdft_norm, Vdftvec_norm,fvecN] = fft_normPower(Vdftave_fld, Vdftave_unfld, freqmat, plots)
end = time.time()
print(end-start)


#### PLOT FIGURES ####

def spec2m(x):
    return(1/x)
    
def m2spec(x):
    return(1/x)
fig1 = plt.figure()
ax1=fig1.add_subplot(1,1,1)
ax1.loglog(fvec,Vdftvec_fld,'r.', label = 'Landslide Terrane')
ax1.loglog(fvec,Vdftvec_unfld,'b.', label = 'Non-landslide')
ax1.set_xlabel('Wavelength')
ax1.set_ylabel('Spectral Power')
#ax1.set_title('Spectra for non-landslide terrain')
ax1.legend()
ax1.set_xlim(0.03, 1)
ax1.set_ylim(10e-11, 10e-1)
secax = ax1.secondary_xaxis('top',functions = (spec2m,m2spec))
secax.set_xlabel('Distance (m)')
fig1.show()

fig2 = plt.figure()
ax2=fig2.add_subplot(1,1,1)
ax2.loglog(fvec,Vdftvec_fld,'r.',label = "Landslide Terrain")
ax2.set_xlabel('Wavelength')
ax2.set_ylabel('Spectral Power')
#ax2.set_title('Spectra for landslide terrain')
ax2.legend()
ax2.set_xlim(0.03, 1)
ax2.set_ylim(10e-11, 10e-1)
secax = ax2.secondary_xaxis('top',functions = (spec2m,m2spec))
secax.set_xlabel('Distance (m)')
fig2.show()






fig2 = plt.figure()
ax2=fig2.add_subplot(1,1,1)
ax2.semilogx(fvecN,Vdftvec_norm,'.')
ax2.set_xlabel('Wavelength')
ax2.set_ylabel('Normalized Power')
#ax2.set_title('Normalized spectral power')
secax = ax2.secondary_xaxis('top',functions = (spec2m,m2spec))
secax.set_xlabel('Distance (m)')
fig2.show()



#%%  ######### Test of CWT Roughness ########
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt')
from conv2_mexh_var import conv2_mexh_var
dem_path = os.path.join(os.getcwd(),'ls_patch_4.tif')
demFLD = gdal.Open(dem_path)
demFLD = np.array(demFLD.GetRasterBand(1).ReadAsArray())

dem_path = os.path.join(os.getcwd(),'no_ls_p4.tif')
demUNfld = gdal.Open(dem_path)
demUNfld = np.array(demUNfld.GetRasterBand(1).ReadAsArray()) #this removes

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_roughness_map_sb')
#from conv2_mexh_var import conv2_mexh_var
#from conv2_mexh import conv2_mexh


dx = 0.5
scales = np.exp(np.linspace(0,4,20))
[Vcwt_fld, frq, wave] = conv2_mexh_var(demFLD, scales, dx)
[Vcwt_unfld, _, _] = conv2_mexh_var(demUNfld, scales, dx)



Vcwt_fld = np.transpose(Vcwt_fld)
Vcwt_unfld = np.transpose(Vcwt_unfld)
Vcwt_norm = Vcwt_fld/Vcwt_unfld
#%% plot results from the wavelet transform
from matplotlib.ticker import FormatStrFormatter
def spec2m(x):
    return(1/x)
    
def m2spec(x):
    return(1/x)

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

#### found ideal wavelength of landslide is 14 - 8.3 m ####
#### THIS CORRESPONDS TO A SCALE OF 7 - 4
#%%
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
from conv2_mexh2 import conv2_mexh2
from ROC_plot import ROC_plot
#This is the step where some work is involved in solving for the wavelet scale needed to filter out the wavelenths of interest
# W = 2*np.pi*dx*s/(np.sqrt(5/2))
#S = (w * np.sqrt(5/2)/(2*np.pi*dx))
fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []
dx = 0.5
###################################################
###################################################
##### THIS CORRESPONDS TO 14 - 46 M ROUGHNESS ####
###################################################
###################################################
[C1,_,_] = conv2_mexh2(DEM, 7,dx)
#[C2,_,_] = conv2_mexh2(DEM, 9,dx)
[C3,_,_] = conv2_mexh2(DEM, 11,dx)
#[C4,_,_] = conv2_mexh2(DEM, 13,dx)
[C5,_,_] = conv2_mexh2(DEM, 15,dx)
#[C6,_,_] = conv2_mexh2(DEM, 17,dx)
[C7,_,_] = conv2_mexh2(DEM, 19,dx)
#[C8,_,_] = conv2_mexh2(DEM, 21,dx)
[C9,_,_] = conv2_mexh2(DEM, 23,dx)
#[C10,_,_] = conv2_mexh2(DEM,25,dx)
# Square and sum wavelet coefficients in quadrature

#Vcwtsum = C1**2 + C2**2 + C3**2  + C4**2 + C5**2 + C6**2  + C7**2 + C8**2 + C9**2 + C10**2
Vcwtsum = C1**2 + C3**2  + C5**2   + C7**2 + C9**2 
del(C1,C3,C5,C7,C9)           

fig1, ax1= plt.subplots()
plt.imshow(np.log(Vcwtsum))
ax1.set_title('CWT Spectral Power Sum')

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)
#%%
#fig2, ax2 = plt.imshow()
#ax2.set_title('CWT Power Sum (smoothed)')
#ax2.imshow(np.log(Vcwt_smooth))

### EXPORT CWT IMAGE #####
#os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_ls_map')
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Report\Figures\Figure_7')
fName = 'CWT'  + '_rghness_14_T_8.tif'
lsMap = np.float32(np.log(Vcwt_smooth))
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)

lowT = 2
highT = 11
numDiv = 40
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve


fName = 'CWT'  + '_lsmap_14_t_8.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
#%% CWT WITH A SLIGHTLY DIFFERENT WAVELENGTH
###################################################
###################################################
##### THIS CORRESPONDS TO 11 - 9 M ROUGHNESS ####
###################################################
###################################################   
dx = 0.5
[C2,_,_] = conv2_mexh2(DEM, 7,dx)
[C3,_,_] = conv2_mexh2(DEM, 5,dx)
[C4,_,_] = conv2_mexh2(DEM, 4,dx)



# Square and sum wavelet coefficients in quadrature

Vcwtsum = C2**2 + C3**2 + C4**2 
del(C2,C3,C4)
fig1, ax1= plt.subplots()
plt.imshow(np.log(Vcwtsum)) 
ax1.set_title('CWT Spectral Power Sum')

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)

#fig2, ax2 = plt.imshow()
#ax2.set_title('CWT Power Sum (smoothed)')
#ax2.imshow(np.log(Vcwt_smooth))

### EXPORT CWT IMAGE #####
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Report\Figures\Figure_7')
fName = 'CWT'  + '_rghness_16_t_8.tif'
lsMap = np.float32(np.log(Vcwt_smooth))
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
# iMPORT To arcgis and check range
    

lowT = -2
highT = 5
numDiv = 40
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve

fName = 'CWT'  + '_lsmap_13_t_7.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
    
#%%
    
#%% CWT WITH A SLIGHTLY DIFFERENT WAVELENGTH
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
fig1, ax1= plt.subplots()
plt.imshow(np.log(Vcwtsum))
ax1.set_title('CWT Spectral Power Sum')

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)

#fig2, ax2 = plt.imshow()
#ax2.set_title('CWT Power Sum (smoothed)')
#ax2.imshow(np.log(Vcwt_smooth))

### EXPORT CWT IMAGE #####
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Report\Figures\Figure_10_Alternate_CWT_Tests')
fName = 'CWT'  + '_rghness_18_t_25.tif'
lsMap = np.float32(np.log(Vcwt_smooth))
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
# iMPORT To arcgis and check range
    

lowT = 0
highT = 8.1
numDiv = 40
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve

fName = 'CWT'  + '_lsmap_18_t_25.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
    
#%%
    
#%% CWT WITH A SLIGHTLY DIFFERENT WAVELENGTH
###################################################
###################################################
##### THIS CORRESPONDS TO 18 - 8.3 M ROUGHNESS ####
###################################################
###################################################   
[C2,_,_] = conv2_mexh2(DEM, 5.5,dx)
[C3,_,_] = conv2_mexh2(DEM, 5,dx)
#[C4,_,_] = conv2_mexh2(DEM, 4.5,dx)
[C5,_,_] = conv2_mexh2(DEM, 8,dx)
[C6,_,_] = conv2_mexh2(DEM, 9,dx)


# Square and sum wavelet coefficients in quadrature

Vcwtsum = C2**2 + C3**2 + C4**2 + C5**2 + C6**2

fig1, ax1= plt.subplots()
plt.imshow(np.log(Vcwtsum))
ax1.set_title('CWT Spectral Power Sum')

radius = 25
from smooth2 import smooth2
[Vcwt_smooth,_] = smooth2(Vcwtsum,radius)

#fig2, ax2 = plt.imshow()
#ax2.set_title('CWT Power Sum (smoothed)')
#ax2.imshow(np.log(Vcwt_smooth))

### EXPORT CWT IMAGE #####
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_ls_map')
fName = 'CWT'  + '_rghness_11_t_9.tif'
lsMap = np.float32(np.log(Vcwt_smooth))
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
# iMPORT To arcgis and check range
    

lowT = -3
highT = 4.5
numDiv = 70
[fpr,tpr, lsMap, tmax] = ROC_plot(lsMap,lsRaster,lowT,highT,numDiv)

auc = np.trapz(tpr,fpr)
fprL.append(fpr) #save outputs for false positive rate
tprL.append(tpr) # save outputs for true positive rate
tMaxL.append(tmax) # save outputs for max threshold cutoff
aucL.append(auc) # save outputs for area under curve

fName = 'CWT'  + '_lsmap_11_t_9.tif'
lsMap = np.float32(lsMap)
with rio.open(fName,'w',**Meta) as dst:
    dst.write(lsMap,1)
#%% Plot all ROCS for CWT
winds = ["12-50 m", "18 - 25 m","7 - 13 m"]
zz = np.vstack([np.asarray(aucL), np.asarray(tMaxL),winds])
zz = zz.T
import pandas as pd

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

#%% DCE EIGEN ROC

# window sizes = 5, 10, 15, 20
os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\wavelets')
from ROC_plot import ROC_plot
from DC_eig_par import DC_eig_par
from DCE_preprocess import DCE_preprocess
import progress as progress

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_ls_map')

fprL = []
tprL = []
lsMap = []
tMaxL = []
aucL = []
wL = []

maxW = 40
minW = 10
inc = 5

for i in range(minW,maxW,inc):
    print(i)
    print()
    [cx,cy,cz] = DCE_preprocess(DEM,cellsize,i)
    eps = np.finfo(float).eps
    eps = np.float32(eps)
    eigGrd = DC_eig_par(DEM, i,cx,cy,cz,eps)
    fName = 'DCE_w' + str(i) + '.tif'
    eigGrd = np.float32(eigGrd)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(eigGrd,1)
        
    lowT = 0
    highT = 1
    numDiv = 70
    [fpr,tpr, lsMap, tmax] = ROC_plot(eigGrd,lsRaster,lowT,highT,numDiv)
    
    auc = np.trapz(tpr,fpr)
    fprL.append(fpr) #save outputs for false positive rate
    tprL.append(tpr) # save outputs for true positive rate
    tMaxL.append(tmax) # save outputs for max threshold cutoff
    aucL.append(auc) # save outputs for area under curve
    wL.append(i) # save window size outputs

    plt.plot(fpr,tpr)
    
    fName = 'DCE_w' + str(i) + '_lsMap.tif'
    lsMap = np.float32(lsMap)
    with rio.open(fName,'w',**Meta) as dst:
        dst.write(lsMap,1)

zz = np.vstack([np.asarray(wL), np.asarray(aucL), np.asarray(tMaxL)])
zz = zz.T
import pandas as pd

os.chdir(r'U:\GHP\Projects\NSF - Morris Landslides\Code\Developmemnt\testing_roughness_map_sb')
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

figName = 'DCE_ROC_Curves.pdf'
plt.savefig(figName)