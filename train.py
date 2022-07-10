import cv2
import os
import numpy as np


def loadImages(hog, path, arr):
    file_list = os.listdir(path)
    for filename in file_list:
        img = cv2.imread(path + filename)
        img = cv2.resize(img, (64,64))
        hist = hog.compute(img)
        arr.append(hist)

winSize = (64, 64)
blockSize = (32, 32) 
blockStride = (16, 16)
cellSize = (8, 8)
nBin = 9 

# histogram of gradients
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)

path = "C:/Users/ASUS/3D Objects/Internship_Project/FullIJCNN2013/"

positives = []
negatives = []
data = os.listdir(path)
for x in data:
    if "." not in x and ( len(x) == 1 or len(x) == 2): #positive images
        loadImages(hog, path+x+"/", positives)
    elif ".ppm" in x: #negative images
        img = cv2.imread(path + x)
        patch = img[0:64, 0:64, :]
        hist = hog.compute(patch)
        negatives.append(hist)

print("beginning train....")
featuresPositive = np.array(positives)
featuresNegative = np.array(negatives)

labelsPositive = np.ones(((len(featuresPositive))), np.int32)
labelsNegative = np.zeros(((len(featuresNegative))), np.int32)

features = np.float32(np.concatenate((featuresPositive, featuresNegative), axis=0))
labels = np.concatenate((labelsPositive, labelsNegative), axis=0)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))


ret = svm.train(features, cv2.ml.ROW_SAMPLE, labels)
svm.save('traffic-sign.xml')
print("saved: traffic-sign.xml")