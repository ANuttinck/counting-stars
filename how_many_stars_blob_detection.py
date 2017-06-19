# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:16:38 2017

@author: Antoine
"""

import cv2
import numpy as np

# Read image
filename = 'D:\Antoine\Documents\Stage\Partoo\star_wars.png'
im = cv2.imread(filename)  # [:100,:200]

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Threshold parameters
params.minThreshold = 5
params.maxThreshold = 30
params.thresholdStep = 0.5
params.minDistBetweenBlobs = 5

# Color filter parameters
params.filterByColor = 1
params.blobColor = 255

# Size filter parameters
params.filterByArea = 1
params.minArea = 0.5
params.maxArea = 15

# Shape filter parameters
params.filterByConvexity = 1
params.minConvexity = 0.9
# params.filterByCircularity = 0
# params.filterByInertia = 0

# Set up the detector with these parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(im)

# print("Threshold step: ", params.thresholdStep)
for param in params.__dir__():
    if param[:2] != "__":
        print(param, str(getattr(params, param)))
print("Number of keypoints: ", len(keypoints))

# Draw detected blobs as red circles
 # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()