import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression

import read




mainFolder = "FullIJCNN2013//"
dataset, file_list, annotation = read.read(mainFolder)

img = cv2.imread(mainFolder + file_list[11])
                 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
template = dataset[79][1]
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.45
loc = np.where( res >= threshold)

results = []
for pt in zip(*loc[::-1]):
    results.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    
    
for x1,y1,x2,y2 in results:
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    

plt.imshow(img)