import cv2
import numpy as np
import os

winSize = (64, 64)
blockSize = (32, 32) 
blockStride = (16, 16)
cellSize = (8, 8)
nBin = 9 

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)

svm = cv2.ml.SVM_load("traffic-sign.xml")


(rho, alpha, supportVectorIndices) = svm.getDecisionFunction(0)
supportVectors = svm.getSupportVectors()
sv_new = np.append(supportVectors, -rho)

hog.setSVMDetector(sv_new)

path = "C:/Users/ASUS/3D Objects/Internship_Project/FullIJCNN2013/"

files = os.listdir(path)
img_files = list(filter(lambda x: '.ppm' in x, files))
for x in img_files:

    img = cv2.imread(path + x)
    (rects, weights) = hog.detectMultiScale(img)
    for (x,y,w,h) in rects:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.putText(img, "Traffic sign", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
    
    cv2.imshow("img", img)
    ch = cv2.waitKey(0)
    if  ch == ord('q'):
        break


cv2.destroyAllWindows()

