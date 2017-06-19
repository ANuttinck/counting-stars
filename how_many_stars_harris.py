# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:00:35 2017

@author: Antoine
"""

import cv2
import numpy as np

filename = '..\star_wars.png'
# Load an color image + ,0 directly in grayscale
img = cv2.imread(filename)  # [:100,:200,:]
# HARRIS DETECTOR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# resuts is dilated for making the corners not important
# dst = cv2.dilate(dst, None)
# --> More corners founded with a dilated image !!!

# Threshold for an optimal value, it maychange depending on the image
filter_corners = dst>0.01*dst.max()
img[filter_corners] = [0, 0, 255]
print(np.sum(filter_corners))
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()