import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from imutils.object_detection import non_max_suppression
from timeit import default_timer as timer

import read

def detect(img, template, kp1, des1, kp2, des2, threshold):

    h, w, d = img.shape
    img_points = np.zeros((h, w))
    points = []
    
    matches = matcher.knnMatch(des1, des2, 2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i, (m1,m2) in enumerate(matches):
        if m1.distance < 0.5 * m2.distance:
            matchesMask[i] = [1,0]
#            print("matches found")
            ## Notice: How to get the index
            pt1 = kp1[m1.queryIdx].pt
            pt2 = kp2[m1.trainIdx].pt
#            print(i, pt1,pt2 )
            points.append([pt1[0],pt1[1]])

    for i in range(len(points)):
        img_points [int(points[i][1]),int(points[i][0])] = 1
                     
#        h, w, d = template.shape
#
#        loc = np.where( img_points >= 0.5)       
#        for pt in zip(*loc[::-1]):
#            points.append([pt[0],pt[1]])
            
    return img_points, points


mainFolder = "FullIJCNN2013//"
dataset, file_list, annotation = read.read(mainFolder)

f= open("kp-results.txt","w+")
sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
## Create flann matcher
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})


start = timer()
kpList=[]
for name,template in dataset:
    kp2, des2 = sift.detectAndCompute(template,None)
    if des2 is None or len(des2)<5 :
            continue
    kpList.append([name,template,kp2,des2])

end = timer()
print("time to calculate keypoints for dataset: ", end - start) 
print("length of dataset that found kp: ", len(kpList)) 



for i in range(0 , 5 ): #len(file_list)):
    
    start = timer()
    print(file_list[i])
    img = cv2.imread(mainFolder + file_list[i])
#    img = imutils.resize(img, width=min(800, img.shape[1]))
    
    kp1, des1 = sift.detectAndCompute(img,None)
    
    results = []
    for templateName, template, kp2, des2 in kpList:    
        
        image_points, points = detect(img, template.copy(), kp1, des1, kp2, des2, threshold=0.5)
        kp2Count = len(kp2)
        
        if (len(points))<2 or len(points) < (kp2Count*0.6):
            continue
               
#        cv2.imshow("template", template)
#        print(index," - ",  kp2Count, " - ", len(points))
#        cv2.waitKey(0)
        
        imgTemp = img.copy()

        window_height, window_width, d = template.shape
        stepSize = 50 #int (np.min([window_height, window_width]) / 2)
        
        isFound=False
        for y1 in range(0, image_points.shape[0], stepSize):
            for x1 in range(0, image_points.shape[1], stepSize):
                                
                y2 = y1 + window_height
                x2 = x1 + window_width

                points_patch = [ [x,y] for [x,y] in points if y>y1 and y<y2 and x>x1 and x<x2 ]  
                
                if len(points_patch) >= (kp2Count*0.6):
#                    print("found len:", len(points_patch))
                    isFound = True
                    results.append([x1, y1, x2, y2])
#                    cv2.rectangle(imgTemp, (x1,y1), (x2,y2), (0,0,255), 3) 
                    print("found for template : ",templateName," coors",x1,y1)

#        if isFound:
#            cv2.imshow("resTemp",cv2.resize(imgTemp,(448,448)))
#            cv2.imshow("template", template)
#            cv2.imshow("image_points",cv2.resize(image_points,(448,448)))
#            cv2.waitKey(0)
                    
    rects = np.array([[x1,y1,x2,y2] for (x1, y1, x2, y2) in results])  
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.12)  
    for x1,y1,x2,y2 in pick:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)  
        f.write(file_list[i]+";"+str(x1)+";"+str(y1)+";"+str(x2)+";"+str(y2)+"\n")
            
    cv2.imshow("results", cv2.resize(img,(448,448)))
    
    end = timer()
    print("elapsed time: ", end-start)
    print("results: ", pick)
#    break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
#    if (len(results)>0):
#        cv2.waitKey(0)
        
        
f.close()

