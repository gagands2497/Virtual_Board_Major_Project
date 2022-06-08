import cv2
import mediapipe as mp
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detection
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.results = None
        self.mpHands = mp.solutions.hands     
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils  
        
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        self.results = self.hands.process(imgRGB)  
        if self.results.multi_hand_landmarks:   
            for handLms in self.results.multi_hand_landmarks:   
                if draw:
                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)    # joining points on our hand
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        if self.results.multi_hand_landmarks:   
            myHand = self.results.multi_hand_landmarks[handNo]  
            for id, lm in enumerate(myHand.landmark): 
                h, w, c = img.shape    
                cx, cy = int(lm.x*w), int(lm.y*h)   
               
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmlist, bbox

    def fingersUp(self):    
        fingers = []   
        
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
       
            fingers.append(1)
        else:
            fingers.append(0)
       
        for id in range(1,  5):   
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3): 
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]     
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]    
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2     

        if draw:    
            cv2.line(img, (x1, y1), (x2, y2), (125, 125, 125), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def multi_distance(self, pts, img, draw=True, r=15, t=3):
        color = [0]*3
        x1, y1 = self.lmlist[pts[0]][1], self.lmlist[pts[0]][2]    
        x2, y2 = self.lmlist[pts[1]][1], self.lmlist[pts[1]][2]   
        x3, y3 = self.lmlist[pts[2]][1], self.lmlist[pts[2]][2]     
        y = [y1, y2, y3]
        cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2     
        cx2, cy2 = (x3 + x2) // 2, (y3 + y2) // 2    
        maxy = max(y)
        miny = maxy-255
        for i, u in enumerate(y):
            factor = (u - miny)/(maxy - miny)
            color[i] = factor*255
        if draw:    
            cv2.line(img, (x1, y1), (x2, y2), (125, 125, 125), t)
            cv2.line(img, (x2, y2), (x3, y3), (125, 125, 125), t)
            cv2.circle(img, (cx1, cy1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx2, cy2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (0, 0, color[0]), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, color[1], 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), r, (color[2], 0, 0), cv2.FILLED)
        color = tuple(i for i in color)
        return color