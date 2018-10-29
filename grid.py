import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,9))

def detect_grid(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_line = np.array([12,0,128])
    upper_line = np.array([45,191,255])

    mask = cv2.inRange(hsv,lower_line,upper_line)
    # res = cv2.bitwise_and(frame,frame,mask=mask)

    median = cv2.medianBlur(mask, 5)
    dilation = cv2.dilate(median,kernel,iterations = 1)


    cv2.imshow('frame',frame)
    cv2.imshow('mask',dilation)
    # cv2.imshow('res',res)

    cv2.imwrite('grid_line.png', dilation)

    key = cv2.waitKey(0)



frame = cv2.imread('frame_0040.png')
detect_grid(frame)