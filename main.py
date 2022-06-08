import cv2
import handtrackingmodule as htm
import numpy as np
import os

overlayList = []  
Thickness = 5 
drawColor = (255, 0, 255)   
xp, yp = 0, 0
height, width = 720, 1280
# images in header folder
folderPath = "Header/new"
toolList = os.listdir(folderPath)    
# ------------------------------------------------- #

for imPath in toolList:   
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)  
header = overlayList[0]    

cap = cv2.VideoCapture(0)
cap.set(3, width)  
cap.set(4, height)  
imgCanvas = np.zeros((height, width, 3), np.uint8)     
detector = htm.handDetector(detectionCon=0.85, maxHands=1) 

while True:
    
    success, img = cap.read()
    img = cv2.flip(img, 1)  

    img = detector.findHands(img)   
    lmList, bbox = detector.findPosition(img, draw=False)
   
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]    
        x2, y2 = lmList[12][1:]   
        
        fingers = detector.fingersUp()
        
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if x1 < 150:
                if (height*0.2) < y1 < (height*0.4): 
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif (height*0.4) < y1 < (height*0.6):   
                    header = overlayList[1]
                    drawColor = (255, 100, 10)
                elif (height*0.6) < y1 < (height*0.7):    
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif (height*0.7) < y1 < height:  
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                elif 0 < y1 < (height*0.2):
                    imgCanvas = np.zeros((height, width, 3), np.uint8)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            
        elif fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)   
            if xp == 0 and yp == 0:
                
                xp, yp = x1, y1 
            
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, Thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, Thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, Thickness)
                
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, Thickness)
            xp, yp = x1, y1    
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    
    
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)   
    

    img = cv2.bitwise_and(img, imgInv)
    
    
    img = cv2.bitwise_or(img, imgCanvas)

    
    img[0:720, 0:150] = header  

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break