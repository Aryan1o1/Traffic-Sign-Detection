#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
#%%


sift = cv2.xfeatures2d.SIFT_create()

#%%

## Create flann matcher
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})



img1 = plt.imread("image1.jpg")
img2 = plt.imread("patch.jpg")


kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


matches = matcher.knnMatch(des1, des2, 2)


# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.1 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


plt.imshow(img3),plt.show()
#plt.imshow()


#%%




