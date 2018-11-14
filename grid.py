import cv2
import numpy as np
# from numpy import linalg as LA
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
mark_area_min = 40
mark_area_max = 150
mark_ratio_min = 0.75

top_left_mark = None
bottom_left_mark = None

# 底下是 find_layout 參數
extend_dist = 30
th_overlap = 100
img_w = None
img_h = None

layout_map = np.full((15,6), -1)
grid_idx_map = None


class Cell:
    ''' x1    x4
        x2    x3'''
    def __init__(self, cen, con):
        self.center = cen # [x, y]
        self.polygon = con # np.array [x, y]
        self.col = -1
        self.row = -1
        self.coord = None
        self.find_corners()

    def find_corners(self):
        min_dist = 999999
        for i, cc in enumerate(self.polygon):
            cx = cc[0]
            cy = cc[1]
            dist = cx**2 + cy **2
            if dist < min_dist:
                self.x1 = cc # top_left
                min_dist = dist        

        min_dist = 999999
        for i, cc in enumerate(self.polygon):
            cx = cc[0]
            cy = img_h - cc[1]
            dist = cx**2 + cy **2
            if dist < min_dist:
                self.x2 = cc # bottom_left
                min_dist = dist    

        min_dist = 999999
        for i, cc in enumerate(self.polygon):
            cx = img_w - cc[0]
            cy = img_h - cc[1]
            dist = cx**2 + cy **2
            if dist < min_dist:
                self.x3 = cc # bottom_right
                min_dist = dist   

        min_dist = 999999
        for i, cc in enumerate(self.polygon):
            cx = img_w - cc[0]
            cy = cc[1]
            dist = cx**2 + cy **2
            if dist < min_dist:
                self.x4 = cc # top_right
                min_dist = dist   


    def set_coord(self, coord):
        self.coord = coord


def find_top_left_mark(landmark_center_lst):
    global top_left_mark
    global bottom_left_mark

    min_dist = 999999
    for i, cc in enumerate(landmark_center_lst):
        cx = cc[0]
        cy = cc[1]
        if cx**2 + cy **2 < min_dist:
            top_left = i
            min_dist = cx**2 + cy **2

    min_dist = 999999
    for i, cc in enumerate(landmark_center_lst):
        cx = cc[0]
        cy = img_h - cc[1]
        if cx**2 + cy **2 < min_dist:
            bottom_left = i
            min_dist = cx**2 + cy **2

    top_left_mark = landmark_center_lst[top_left]
    bottom_left_mark = landmark_center_lst[bottom_left]

    print('top_left {}({},{}), bottom_left {}'.format(top_left, 
        top_left_mark[0],
        top_left_mark[1],
        bottom_left))

def move_down(curr_idx, cell_list):
    cell_mask = np.zeros(grid_idx_map.shape, dtype=int)
    # print(cell_list[i].polygon)

    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[1][1] += extend_dist
    cell_ext[2][1] += extend_dist
    # print(cell_ext)

    cv2.drawContours(cell_mask, [cell_ext], -1, (255,255,255), -1)
    next_mask = cell_mask & grid_idx_map 
    next_mask = next_mask[next_mask!=curr_idx+1]
    next_mask = next_mask[next_mask!=0]
    # cv2.imwrite('next_mask.png', next_mask)
    nonzero = np.count_nonzero(next_mask)
    if  nonzero < th_overlap:
        next_idx = -1
    else:
        next_idx = int(np.median(next_mask))-1

    print(next_idx, nonzero)
    return next_idx

def move_right(curr_idx, cell_list):
    cell_mask = np.zeros(grid_idx_map.shape, dtype=int)
    # print(cell_list[i].polygon)

    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[2][0] += extend_dist
    cell_ext[3][0] += extend_dist
    # print(cell_ext)

    cv2.drawContours(cell_mask, [cell_ext], -1, (255,255,255), -1)
    next_mask = cell_mask & grid_idx_map 
    next_mask = next_mask[next_mask!=curr_idx+1]
    next_mask = next_mask[next_mask!=0]
    # cv2.imwrite('next_mask.png', next_mask)
    nonzero = np.count_nonzero(next_mask) 
    if nonzero < th_overlap:
        
        next_idx = -1
    else:
        next_idx = int(np.median(next_mask))-1

    print(next_idx, nonzero)
    return next_idx

def find_layout(landmark_center_lst, cell_list):
    global layout_map

    min_dist = 999999    
    for i, cell in enumerate(cell_list):
        # print(i, cell.top_left)
        dx = cell.x1[0]-top_left_mark[0]
        dy = cell.x1[1]-top_left_mark[1] 

        dist = dx**2 + dy **2
        if dist < min_dist:
            min_dist = dist
            first_cell = i

    print('first cell ', first_cell)

    # cell_mask = np.zeros(grid_idx_map.shape, dtype=int)
    layout_map[0, 0] = first_cell
    col = 0
    num_rows = -1
    while True:
        i = first_cell
        row = 0
        while True:
            next_idx = move_down(i, cell_list)
            if next_idx < 0:
                print('break, move_down count_nonzero <', th_overlap)
                break

            if num_rows ==-1 :
                if cell_list[next_idx].x1[1] +5 > bottom_left_mark[1]:
                    break
            elif row+1 >  num_rows:
                break 

            row += 1
            # print(next_idx)
            layout_map[row,col] = next_idx
            cell_list[i].set_coord((row,col))
            i = next_idx

        if col==0:
            num_rows = row

        next_col = move_right(first_cell, cell_list)
        if next_col < 0:
            print('break, move_right count_nonzero <', th_overlap)
            break

        col += 1
        first_cell = next_col
        layout_map[0,col] = next_col
        cell_list[next_col].set_coord((0,col))


    print(layout_map)

def detect_landmark(small, logging, debug=False):
    # small = cv2.pyrDown(frame)
    img_b, img_g, img_r = cv2.split(small) 
    th, bin_b = cv2.threshold(img_b, th_mark, 255, cv2.THRESH_BINARY) 
    th, bin_g = cv2.threshold(img_g, th_mark, 255, cv2.THRESH_BINARY)
    th, bin_r = cv2.threshold(img_r, th_mark, 255, cv2.THRESH_BINARY)
    landmark = bin_b & bin_g #& bin_r
    median = cv2.medianBlur(landmark, 3)
    landmark = cv2.dilate(median, landmark_kernel, iterations = 1)

    # if debug:
    #     cv2.imshow('landmark',landmark)
    #     # cv2.imshow('bin_color',bin_color)
    #     key = cv2.waitKey(0)    

    _, contours, hierarchy = cv2.findContours(landmark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    landmark_list = []
    landmark_center_lst = []    
    for con in contours:
        area = cv2.contourArea(con)
        if area < mark_area_min or area > mark_area_max:
            continue

        # clen = len(con)
        # compact = (clen * clen) / area;
        # # print('clen:', clen, compact)
        # if compact > cell_compact_th:
        #     continue

        rect = cv2.minAreaRect(con)
        width, height = rect[1]
        angle = rect[2]
        if width > height:
            ratio = height/width
        else:
            ratio = width/height
        # print('ratio ', ratio)
        if ratio < mark_ratio_min:
            continue

        M = cv2.moments(con)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # print(cx, cy)
        landmark_center_lst.append([cx,cy])

        # approx = cv2.approxPolyDP(con, 3,True)
        # landmark_list.append(approx)

    return landmark_center_lst
     




def detect_grid(small, cell_list, logging, debug=False):
    global grid_idx_map

    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,th_block_size,th_c)
    bin_img = cv2.dilate(bin_img, grid_kernel, iterations = 1)
    bin_img = 255 - bin_img

    _, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # if debug:
    #     bin_color = cv2.cvtColor(bin_img,cv2.COLOR_GRAY2BGR)
    #     cv2.drawContours(bin_color, contours, -1, (0,255,0), 1)
    
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
        # center_lst.append([cx,cy])
        approx = cv2.approxPolyDP(con, polygon_dist,True)
        # print('approx nodes:', len(approx))
        if len(approx) != 4:
            print('X approx nodes:', len(approx))
            continue

        approx_arr = np.array(approx).reshape(-1, 2)

        # print('approx_arr ',approx_arr)
        cell = Cell([cx,cy], approx_arr)
        cell_list.append(cell)
        
    grid_idx_map = np.full(bin_img.shape, 0)
    for i, cc in enumerate(cell_list):
        cv2.drawContours(grid_idx_map, [cc.polygon], -1, (i+1,i+1,i+1), -1)
    cv2.imwrite('grid_idx_map.png', grid_idx_map)


 

def identify_grid(frame, logging, debug=False):
    global img_w
    global img_h

    cell_list = []

    small = cv2.pyrDown(frame)
    img_w = small.shape[1]
    img_h = small.shape[0]
    print('small image size (w, h): ', img_w, img_h)

    detect_grid(small, cell_list, logging, debug)

    landmark_center_lst = detect_landmark(small, logging, debug)
    find_top_left_mark(landmark_center_lst)  
    find_layout(landmark_center_lst, cell_list)

    for i, cc in enumerate(cell_list):
        cv2.drawContours(small, [cc.polygon], -1, (0,255,0), 1)
        strText = '{}'.format(i)
        cv2.putText(small, strText, (cc.center[0], cc.center[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.circle(small,(cc.center[0], cc.center[1]),2,(0,0,255),3)
    # cv2.drawContours(small, landmark_list, -1, (0,0, 255), 1)

    for i, cc in enumerate(landmark_center_lst):
        strText = '[{}]'.format(i)
        cv2.putText(small, strText, (cc[0], cc[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0, 255))

    if debug:
        cv2.imshow('cell_list',small)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)


    cv2.imwrite('grid_detection.png', small)




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
        

        identify_grid(frame, logging, args.show_debugmsg)