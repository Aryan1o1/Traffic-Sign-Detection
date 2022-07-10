import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import imutils
import read

def detect(img, template):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    results = []
    for pt in zip(*loc[::-1]):
        results.append([pt[0],pt[1],pt[0] + w, pt[1] + h])
        
    return results

mainFolder = "FullIJCNN2013//"
dataset, file_list, annotation = read.read(mainFolder)


for name, template in dataset:

    img = cv2.imread(mainFolder + file_list[2])
    img = imutils.resize(img, width=min(800, img.shape[1]))
    
    results = detect(img, template)
    
    rects = np.array([[x1,y1,x2,y2] for (x1, y1, x2, y2) in results])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.2)
    
    for x1,y1,x2,y2 in pick:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 4)
    
    