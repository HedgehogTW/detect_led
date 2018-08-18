import cv2
import numpy as np
from datetime import datetime
import timeit

bright_th = 50

def find_blobs(img_bin):
    # np.copyto(bb, img_bin)
    con_candidate = []
    areaLower = 10
    areaUpper = 50
    _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite('test.jpg',img_bin)
    for con in contours:
        area = cv2.contourArea(con)
        if areaLower < area < areaUpper:
            con_candidate.append(con)
#            print(con)            
    return con_candidate


img = cv2.imread('red_2_small.jpg')
img_b, img_g, img_r = cv2.split(img)  
th, img_bin = cv2.threshold(img_b,bright_th, 255,cv2.THRESH_BINARY)
# img_bin = cv2.medianBlur(img_bin,7)
# img_binmor = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
print(timeit.timeit(lambda: find_blobs(img_bin), number=100)/100)

