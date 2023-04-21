#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:34:09 2021

@author: compraka
"""
import cv2
import matplotlib
from matplotlib import pyplot as plt
from roipoly import RoiPoly
import numpy as np
from scipy import ndimage, interpolate
import matplotlib.path as path
import pandas as pd

#Have to run from command line. SPyder IDE breaks with roipoly function.
#Save the coordinates as npy file and reload in the in polygon function
#Have not found a good workaound for this right now. 

#Load a frame from the video and draw the roi and obtain the coordinates 
img = cv2.imread('img0304.png')
    
#Execute in terminal 'python3 -m detect_ROI.py' 
#Mark 9 ROI's or zones
#Mark shelter
    
# show image
#plt.use("TkAgg")
plt.imshow(img)
plt.title('Draw helping zone, double-click to end.')
region = RoiPoly(color='r')
xcoords = np.asarray(region.x)
np.save("region_x", xcoords)
ycoords = np.asarray(region.y)
np.save("region_y", ycoords)
        
plt.imshow(img)
region.display_roi()
plt.title('Draw helping zone, double-click to end.')
plt.show(block=False)

        #plt.savefig("activation_zones.png")
