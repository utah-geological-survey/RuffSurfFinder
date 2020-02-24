"""
This code was written by Matthew Morriss to plot a hillshade, or shaded relief map, of a numpy grid which is a Digital Elevation Model, imported into Python through GDAL.




INPUT:
    A numpy array which represents elevation data.


OUTPUT:
    a numpy array ready for visualization with matplotlib.imshow
    
#### EXAMPLE RUN ####
    
    hs_array = hillshade(DEM)
    
    #Plot hillshade
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(hs_array,cmap='Greys', alpha = 0.9)
    ax.plot() 
    
CREDIT:
    This code was modified from code written by rveciana on GitHub. 
    https://github.com/rveciana/geoexamples/tree/master/python/shaded_relief

"""


from numpy import gradient
from numpy import pi
from numpy import arctan
from numpy import arctan2
from numpy import sin
from numpy import cos
from numpy import sqrt


def hillshade(array):
    azimuth = 315
    angle_altitude = 45
    
    x, y = gradient(array)
    slope = pi/2. - arctan(sqrt(x*x + y*y))
    aspect = arctan2(-x, y)
    azimuthrad = azimuth*pi / 180.
    altituderad = angle_altitude*pi / 180.
     
 
    shaded = sin(altituderad) * sin(slope)\
     + cos(altituderad) * cos(slope)\
     * cos(azimuthrad - aspect)
     
    return(255*(shaded + 1)/2)


