import cv2
import numpy as np
import matplotlib.pyplot as plt

#IMAGE_H = 200
#IMAGE_W = 100
img = cv2.imread('2.png') # Read the test img
IMAGE_H, IMAGE_W, C = img.shape
src = np.float32([[300, 451], [395, 379], [487, 379], [582, 451]])
#for x in range (0,4):
    #cv2.circle(img,(src[x][0],src[x][1]),5,(0,0,255),cv2.FILLED)
dst = np.float32([[0, IMAGE_H], [0, 0], [IMAGE_W, 0], [IMAGE_W, IMAGE_H]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


#img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
#cv2.imshow("a",warped_img)
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()