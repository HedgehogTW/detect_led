#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:04:18 2018

@author: cclee
"""

import cv2 
import glob
import numpy as np  
import os
import time
import pathlib

def main():

    cap = cv2.VideoCapture(0)
    bOpenVideo = cap.isOpened()
    if bOpenVideo == False:
        print('Open Video failed')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    print('begin write video, fps = %d, w %d, h %d' % (fps, width, height))
        

    outName = 'video_{}x{}_{}fps.mp4'.format(width, height, fps)
    vidw = cv2.VideoWriter(outName, cv2.VideoWriter_fourcc('XVID'),  ####XVID'),  # *'H264'
                           fps, (width, height), True)  # Make a video

    counter = 0
    while True:
        bVideoRead, frame = cap.read()
        if bVideoRead==False:
            print('cap.read() error, frame:', i)
            break
        vidw.write(frame)
        
        counter += 1
        if counter > 100:
            break
        
        key = cv2.waitKey(100) & 0xFF 
        print(counter, end = ' ')
        if key == 27:
            break
        elif key == 32:
            break 
  
    vidw.release()   
    print('write video:',outName)

if __name__ == '__main__':
    main()