#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import mediapipe as mp
import time
import pycaw
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math


# In[15]:


cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpdraw=mp.solutions.drawing_utils
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    cx_4, cy_4, cx_8, cy_8 = None, None, None, None
    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmarks.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==4:
                    cx_4,cy_4=cx,cy
                elif id==8:
                    cx_8,cy_8=cx,cy
                if cx_4 is not None and cy_4 is not None and cx_8 is not None and cy_8 is not None:
                    cv2.circle(img, (cx_4, cy_4), 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (cx_8, cy_8), 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (cx_4, cy_4), (cx_8, cy_8), (255, 0, 255), 3) 
                    l1,l2=(cx_4+cx_8)//2,(cy_4+cy_8)//2
                    cv2.circle(img, (l1,l2), 15, (255, 0, 255), cv2.FILLED)
                    length=math.hypot(cx_8-cx_4,cy_8-cy_4)
                    vol=np.interp(length,[50,300],[minVol,maxVol])
                    volume.SetMasterVolumeLevel(vol,None)
                    
                    if length<50:
                        cv2.circle(img,(l1,l2),15,(0,255,255),cv2.FILLED)
                    
                
                
            mpdraw.draw_landmarks(img,handlandmarks,mpHands.HAND_CONNECTIONS)
       
        
    cv2.imshow("img",img)
    
    if(cv2.waitKey(1)==27):
        break
cap.release()
cv2.destroyAllWindows()





