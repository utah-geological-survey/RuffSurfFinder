# -*- coding: utf-8 -*-


def ROC_plot(RMSH, lsRaster, lowT, highT,numDiv):
    """Created on Tue Oct 22 11:53:53 2019
    This is a function written to calculate the ROC curve for a terrane roughness map and a boolean map of landslide (0 = not landslide; 1 = mapped landslide). This function was heavily modified and ported from a matlab version developed by Matteo Berti.
    
    Args:
    -----------
        - stdGrd - Roughness map
        - lsRaster - Landslide Boolean Map
        - Range for the threshold of roughness, ##### best to look at the Roughness map in a GIS and then determine these values. ####
        - numDiv - number of divisions between high and low thresholds.
        
    Returns:
        - Plot of the ROC Curve
        - FPr and TPr for the curve plotting
        - map of potential landslides
        
    EXAMPLE:
        lowT = 0 #low roughness threshold
        highT = 20 # high roughness threshold
        numDiv = 40 # how many subdivisions between the two
        #call to ROC_plot
        [fpr,tpr, lsMap, tmax] = ROC_plot(stdGrd,lsRaster,lowT,highT,numDiv)
    @author: matthewmorriss
    """
    import numpy as np
    import progress as progress
    import matplotlib.pyplot as plt
    
    thresh = np.linspace(lowT,highT,numDiv)
    fpr = np.empty([np.size(thresh),1])
    tpr = np.empty([np.size(thresh),1])
    
    for i in np.arange(0,np.size(thresh)):
        total  = np.size(thresh)
        progress.progress(i,total,'Doing long job')
        
        thr = thresh[i]
        TP = np.sum((RMSH >= thr) & (lsRaster == 1))
        
        FP = np.sum((RMSH >= thr) & (lsRaster == 0))
        
        TN = np.sum((RMSH < thr) & (lsRaster == 0))
        
        FN = np.sum((RMSH < thr) & (lsRaster == 1))
    
        fpr[i,0] = FP/(TN+FP)
        tpr[i,0] = TP/(TP+FN)
        
    maxCurve = np.sqrt(np.abs(fpr-tpr)/2)
    vm = np.max(maxCurve)
    idm = np.argwhere(maxCurve == vm)[0]
    
    threshMax = thresh[idm][0]
    xmax = fpr[idm][0] #rates associated with optimum false positive rate
    ymax = tpr[idm][0] # rates associated with optimum true positive rate
    
    lsMap = np.nan*lsRaster
    lsMap[RMSH >= threshMax] = 1
    
    efpr = np.append(fpr,0)
    etpr = np.append(tpr,0)
    
    efpr = np.flipud(efpr)
    etpr = np.flipud(etpr)
    
    efpr = np.append(efpr,1)
    etpr = np.append(etpr,1)
    auc = np.trapz(etpr,efpr)
    
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(efpr,etpr, facecolors = 'none', edgecolors = 'r')
    ax1.plot(efpr,etpr)
    ax1.plot([0, 1], [0,1],'r-.')
    ax1.scatter(xmax,ymax)
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title("ROC Curve, AUC = %1.3f," %auc + " Cutoff = %1.2f" %threshMax)

    
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(fpr,maxCurve)
    ax2.set_xlabel('False Positive rate')
    ax2.set_ylabel('Maximized ROC')
    
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.imshow(lsMap)
    


    return(efpr,etpr, lsMap, threshMax)
    
    
    