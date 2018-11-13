import cv2
import numpy as np
import argparse
import logging
import os, sys  
import time  
from time import localtime, strftime 
# import pathlib

# 底下是 detect_grid 參數
grid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
landmark_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
th_block_size = 33
th_c = -20
cell_area_min = 1200
cell_area_max = 10000
cell_ratio_min = 0.15
cell_ratio_max = 0.55
cell_compact_th = 50
polygon_dist = 10
nBins = 12

# 底下是 detect_landmark 參數
th_mark = 240


def detect_landmark(frame, logging, debug=False):
    small = cv2.pyrDown(frame)
    img_b, img_g, img_r = cv2.split(small) 
    th, bin_b = cv2.threshold(img_b, th_mark, 255, cv2.THRESH_BINARY) 
    th, bin_g = cv2.threshold(img_g, th_mark, 255, cv2.THRESH_BINARY)
    th, bin_r = cv2.threshold(img_r, th_mark, 255, cv2.THRESH_BINARY)
    landmark = bin_b & bin_g #& bin_r
    median = cv2.medianBlur(landmark, 3)
    landmark = cv2.dilate(median, landmark_kernel, iterations = 1)

    if debug:
        cv2.imshow('landmark',landmark)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)    


def detect_grid(frame, logging, debug=False):
    small = cv2.pyrDown(frame)
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    # lower_line = np.array([12,0,128])
    # upper_line = np.array([45,191,255])
    # mask = cv2.inRange(hsv,lower_line,upper_line)
    # res = cv2.bitwise_and(frame,frame,mask=mask)


    bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,th_block_size,th_c)
    bin_img = cv2.dilate(bin_img, grid_kernel, iterations = 1)
    bin_img = 255 - bin_img


    _, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if debug:
        bin_color = cv2.cvtColor(bin_img,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(bin_color, contours, -1, (0,255,0), 1)

    cell_list = []
    center_lst = []    
    for con in contours:
        area = cv2.contourArea(con)
        if area < cell_area_min or area > cell_area_max:
            continue

        clen = len(con)
        compact = (clen * clen) / area;
        # print('clen:', clen, compact)
        if compact > cell_compact_th:
            continue

        # rect = cv2.minAreaRect(con)
        # width, height = rect[1]
        # angle = rect[2]
        # ratio = height/width
        # if ratio < cell_ratio_min or ratio > cell_ratio_max:
        #     continue

        M = cv2.moments(con)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # print(cx, cy)
        center_lst.append([cx,cy])

        approx = cv2.approxPolyDP(con, polygon_dist,True)
        cell_list.append(approx)
        

    # print(center_lst)

    cell_center_arr = np.array(center_lst)
    cell_xx = cell_center_arr[:,0]
    # print(cell_arr, cell_arr.shape)
    histo, bin_edges  = np.histogram(cell_xx, bins=nBins, density=True)
    bin_edges = list(map(int, bin_edges))

    print(histo)
    print(bin_edges)

    ncols = 0
    epsilon = 0.00001
    high_level = False
    col_edges = []
    for i in range(nBins):
        if histo[i] > epsilon:
            if not high_level:
                high_level = True
                ncols += 1
                start = bin_edges[i]
        else:
            if high_level:
                end = bin_edges[i]
                col_edges.append((start, end, (start+end)//2))              
            high_level = False

    if high_level:
        end = bin_edges[nBins]
        col_edges.append((start, end, (start+end)//2))         


    # print('ncols:', ncols)
    # print(col_edges)

    cell_col_idx = []
    for x in cell_xx:
        dmin = 99999
        for idx, cc in enumerate(col_edges):
            dx = abs(x - cc[2])
            if dx < dmin:
                dmin = dx
                cidx = idx
        cell_col_idx.append(cidx)

    col_cell_center_lst = []
    coord_dict = {}
    for i in range(ncols):
        col_cell_center = [ (tuple(cc), k) for k, cc in enumerate(cell_center_arr) if cell_col_idx[k]==i]
        col_cell_center.sort(key=lambda x: x[0][1])
        col_cell_center_lst.append(col_cell_center)


        for row, cc in enumerate(col_cell_center):
            coord_dict[cc[1]] = (row, i)
            strText = '[{},{}]'.format(row, i)
            cv2.putText(small, strText, cc[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

    # print(coord_dict)




    for ll in col_edges:
        bx1 = ll[0]
        bx2 = ll[1]
        bc = ll[2]
        cv2.line(small, (bx1, 0), (bx1, small.shape[0]), (0, 0, 255), 2)
        cv2.line(small, (bx2, 0), (bx2, small.shape[0]), (0, 0, 255), 2)
        cv2.line(small, (bc, 0), (bc, small.shape[0]), (0, 255, 255), 2)


    # print('find {} cells'.format(len(cell_list)))

    cv2.drawContours(small, cell_list, -1, (0,255,0), 1)
    if debug:
        cv2.imshow('cell_list',small)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)


    cv2.imwrite('grid_detection.png', small)


    return coord_dict, cell_list, center_lst

if __name__ == "__main__":
    ini_show_debugmsg = True
    ini_show_image = True

    parser = argparse.ArgumentParser(description='grid')   
    parser.add_argument('--show_image', action="store_true", dest='show_image', default=ini_show_image, help='show debug image')
    parser.add_argument('--noshow_image', action="store_false", dest='show_image', default=ini_show_image, help='no show debug image')
    parser.add_argument('--show_debugmsg', action="store_true", dest='show_debugmsg', default=ini_show_debugmsg, help='show debug message')
    parser.add_argument('--noshow_debugmsg', action="store_false", dest='show_debugmsg', default=ini_show_debugmsg, help='no show debug message')
    parser.add_argument('-f', action="store", dest='video_name', default='gridled_1(1).h264', help='input video file name')
    args = parser.parse_args()


    logpath = os.path.dirname('log/') 
    if not os.path.exists(logpath):
        os.makedirs(logpath)
        print('no output log path, create one') 

    fname = strftime("%Y-%m-%d-%H%M%S", localtime())
    fname += '.log'
    logging_file = os.path.join(logpath, fname)


    print("Logging to", logging_file)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename = logging_file,
        filemode = 'w',
    )
    logging.info("Start of the grid detection")

    print('read grid video:', args.video_name)
    cap = cv2.VideoCapture(args.video_name)
    bOpenVideo = cap.isOpened()
    print('Open grid video: {0} '.format(bOpenVideo))
    if bOpenVideo == True:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('width {}, height {} fps {}'.format(width, height, fps))
        for i in range(10):
            bVideoRead, frame = cap.read()  
        cap.release()

        fnlst = args.video_name.rsplit('.',1) 
        fname = fnlst[0] + '_sample.jpg'

        cv2.imwrite(fname, frame)

        # frame = cv2.imread('grid_img.png')
        detect_landmark(frame, logging, args.show_debugmsg)

        detect_grid(frame, logging, args.show_debugmsg)