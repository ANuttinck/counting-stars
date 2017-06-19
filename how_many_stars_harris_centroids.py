# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:00:35 2017

@author: Antoine
"""

import cv2
import numpy as np

filename = 'D:\Antoine\Documents\Stage\Partoo\star_wars.png'
# Load an color image + ,0 directly in grayscale
img = cv2.imread(filename)  # [:100,:200,:]
# HARRIS DETECTOR WITH SUBPIXEL ACCURACY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# resuts is dilated for making the corners not important
dst = cv2.dilate(dst, None)
# --> Less centroids founded with a dilated image !!!
ret, dst = cv2.threshold(dst, 0.001*dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS +  cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1, -1), criteria)

# draw the centroids of the corners
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:,1], res[:,0]] = [0, 0, 255]
# img[res[:,3], res[:,2]] = [0, 255, 0]

print("Number of centroids", centroids.shape)
print("Number of corners", corners.shape)
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()