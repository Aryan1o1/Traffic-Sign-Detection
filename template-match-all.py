import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from imutils.object_detection import non_max_suppression
from timeit import default_timer as timer

import read 


classNames = {}
classNames["00"], classNames["01"], classNames["02"], classNames["03"] = "20", "30", "50", "60"
classNames["04"], classNames["05"], classNames["06"], classNames["07"] = "70", "80", "80*", "100"
classNames["08"], classNames["09"], classNames["10"], classNames["11"] = "120", "No overtaking", "Truck no overtaking", "Intersection"
classNames["12"], classNames["13"], classNames["14"], classNames["15"] = "Sign12", "Give Way", "Stop", "No entry"
classNames["16"], classNames["17"], classNames["18"], classNames["19"] = "Trucks", "No entry", "Danger", "Sharp Left"
classNames["20"], classNames["21"], classNames["22"], classNames["23"] = "Sharp Right", "Winding", "Uneven road", "Slippery"
classNames["24"], classNames["25"], classNames["26"], classNames["27"] = "Road narrow right", "Road work", "Traffic light", "Pedestrian"
classNames["28"], classNames["29"], classNames["30"], classNames["31"] = "Cliddrens", "Bike", "Snow", "Animal"
classNames["32"], classNames["33"], classNames["34"], classNames["35"] = "No entry", "Right", "Left", "Ahead only"
classNames["36"], classNames["37"], classNames["38"], classNames["39"] = "Ahead of right", "Ahead or left", "Keep Right", "Keep Left"
classNames["40"], classNames["41"] = "Roundabout", "No overtaking"

def detect(img, template, method):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,method)
    threshold = 0.9
    loc = np.where( res >= threshold)
    results = []
    for pt in zip(*loc[::-1]):
        results.append([pt[0],pt[1],pt[0] + w, pt[1] + h])
        
    return results



mainFolder = "C:/Users/ASUS/3D Objects/Internship_Project/FullIJCNN2013/"
dataset, file_list, annotation = read.read(mainFolder)

print("length of dataset:", len(dataset))

f = open("template-results.txt","w+")


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

method = eval(methods[1])


for i in range(0, 5):
    
    start = timer()
    print(file_list[i])
    img = cv2.imread(mainFolder + file_list[i])
#    img = imutils.resize(img, width=min(800, img.shape[1]))
    
    classifications = {}
    results = []
    for name, template in dataset:    
        
        results_temp = detect(img.copy(), template, method)
        
        # **** find Detection class with name *******
        x = name[-13:-11]
        for x1, y1, x2, y2 in results_temp:
            classifications[str(x1)+"-"+str(x2)] = x
        # *******************************************

        if (len(results_temp)>0):        
            results.extend(results_temp)
        
#        x,y,d = template.shape
#        img[0:x,0:y,:] = template
        
    
    rects = np.array([[x1,y1,x2,y2] for (x1, y1, x2, y2) in results])  
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.2)  
    i =0
    for x1,y1,x2,y2 in results:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)  
        f.write(file_list[i]+";"+str(x1)+";"+str(y1)+";"+str(x2)+";"+str(y2)+"\n")
        
        #  ****** print Classification result *******
        text = classNames[classifications[str(x1)+"-"+str(x2)]]
        cv2.putText(img, text, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            
    cv2.imshow("results", cv2.resize(img,(448,448)))
    end = timer()
    print("elapsed time: ", end-start)
    print("results: ", pick)

    # plt.imshow(cv2.resize(img,(448,448)))
    # plt.waitforbuttonpress()        
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    if (len(results)>0):
        cv2.waitKey(0)
        
        
f.close()

