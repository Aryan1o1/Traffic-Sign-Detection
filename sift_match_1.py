#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
def detect(img, template, kp1, des1, kp2, des2, threshold):
    
    h,w,d = img.shape
    img_points = np.zeros((h,w))
    points = []
    
    matches = matcher.knnMatch(des1, des2, 2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
            
            pt1 = kp1[m.queryIdx].pt          
            cv2.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
            
            points.append([pt1[0],pt1[1]])
            
    for i in range(len(points)):
        img_points[int(points[i][1]), int(points[i][0])] = 1
        
    return img_points, points
    
#%%
sift = cv2.xfeatures2d.SIFT_create()

#%%
## Create flann matcher
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})


#%%
img1 = plt.imread("image1.jpg")
img2 = plt.imread("patch.jpg")


kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

        
img_points, points = detect(img1, img2, kp1, des1, kp2, des2, 0.1)

#%%
# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


plt.imshow(img_points)




