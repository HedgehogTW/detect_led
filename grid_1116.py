import cv2
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from math import pi
# from numpy import linalg as LA
import argparse
import logging
import os, sys  
import pickle
import time  
from time import localtime, strftime 
from itertools import combinations 
from shapely.geometry import Polygon
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2": # linux
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    ini_disable_picam = False
    ini_show_debugmsg = False
    ini_show_image = False

else: # PC, read video
    ini_disable_picam = True
    ini_show_debugmsg = True
    ini_show_image = True

args = None

# 底下是 picamera 參數
shutter_speed_landmark = 3000
shutter_speed_grid = 50000

# 底下是 detect_grid 參數
grid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
landmark_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
th_block_size = 33
th_c = -1 #-10
cell_area_min = 850
cell_area_max = 10000
cell_ratio_min = 0.12
cell_ratio_max = 0.55
cell_compact_th = 50
polygon_dist = 10


# 底下是 detect_landmark 參數
th_mark = 60 # 240
mark_area_min = 40
mark_area_max = 190
mark_ratio_min = 0.75

# 底下是 find_layout 參數
extend_dist = 30
th_overlap = 100
left_top_dist_gap = 10
img_w = None
img_h = None


# rect_combination, polygon
insideDist_th = 50
polygon_box_dist_th = 20
interior_angle_min = 60
interior_angle_max = 120
aspect_ratio_th = 0.6
map_size = 30 # 要大於2倍 grid
map_size_out = 20 # final output map size

grid_idx_map = None

def find_x1x2x3x4(points_lst):
    ''' x1    x4
        x2    x3'''    

    x, y, w, h = cv2.boundingRect(points_lst)
    # print(x, y, w, h)
    min_dist = 999999
    for i, cc in enumerate(points_lst):
        cx = cc[0]-x
        cy = cc[1]-y
        dist = cx**2 + cy **2
        if dist < min_dist:
            x1_idx = i # top_left
            min_dist = dist        

    min_dist = 999999
    for i, cc in enumerate(points_lst):
        cx = cc[0] -x
        cy = (y+h) - cc[1]
        dist = cx**2 + cy **2
        if dist < min_dist:
            x2_idx = i # bottom_left
            min_dist = dist    

    min_dist = 999999
    for i, cc in enumerate(points_lst):
        cx = (x+w) - cc[0]
        cy = (y+h) - cc[1]
        dist = cx**2 + cy **2
        if dist < min_dist:
            x3_idx = i # bottom_right
            min_dist = dist   

    min_dist = 999999
    for i, cc in enumerate(points_lst):
        cx = (x+w) - cc[0]
        cy = cc[1] -y
        dist = cx**2 + cy **2
        if dist < min_dist:
            x4_idx = i # top_right
            min_dist = dist   

    return x1_idx, x2_idx, x3_idx, x4_idx

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
        x1_idx, x2_idx, x3_idx, x4_idx = find_x1x2x3x4(self.polygon)
        self.x1 = self.polygon[x1_idx]
        self.x2 = self.polygon[x2_idx]
        self.x3 = self.polygon[x3_idx]
        self.x4 = self.polygon[x4_idx]

    def set_coord(self, coord):
        self.coord = coord


def find_top_left_mark(landmark_center_lst):

    top_left, bottom_left, bottom_right, top_right = find_x1x2x3x4(landmark_center_lst)

    top_right_mark = landmark_center_lst[top_right]
    bottom_right_mark = landmark_center_lst[bottom_right]
    top_left_mark = landmark_center_lst[top_left]
    bottom_left_mark = landmark_center_lst[bottom_left]

    msg = 'top_left {}({},{}), top_right {}({},{})'.format(
        top_left, top_left_mark[0], top_left_mark[1],
        top_right, top_right_mark[0], top_right_mark[1])
    logging.info(msg)
    print(msg)

    msg = 'bottom_left {}({},{}), bottom_right {}({},{})'.format(
        bottom_left, bottom_left_mark[0], bottom_left_mark[1],
        bottom_right, bottom_right_mark[0], bottom_right_mark[1])
    logging.info(msg)
    print(msg)

    return top_left_mark, bottom_left_mark, bottom_right_mark, top_right_mark

def calculate_overlap(curr_idx, cell_ext):
    cell_mask = np.zeros(grid_idx_map.shape, dtype=np.uint8)
    cv2.drawContours(cell_mask, [cell_ext], -1, (255,255,255), -1)
    next_mask = cell_mask & grid_idx_map 
    next_mask = next_mask[next_mask!=curr_idx+1]
    next_mask = next_mask[next_mask!=0]
    cv2.imwrite('next_mask.jpg', next_mask)
    nonzero = np.count_nonzero(next_mask)
    if  nonzero < th_overlap:
        next_idx = -1
    else:
        next_idx = int(np.median(next_mask))-1

    return  next_idx, nonzero


def move_up(curr_idx, cell_list):
    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[0][1] -= extend_dist
    cell_ext[3][1] -= extend_dist
    if cell_ext[0][1] < 0:
        cell_ext[0][1] = 0
    if cell_ext[3][1] < 0:
        cell_ext[3][1] = 0
    # print(cell_ext)

    next_idx, nonzero = calculate_overlap(curr_idx, cell_ext)
    msg = 'move_up: {} -> {}, nonzero {}'.format(curr_idx, next_idx, nonzero)
    logging.info(msg)
    return next_idx

def move_down(curr_idx, cell_list):
    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[1][1] += extend_dist
    cell_ext[2][1] += extend_dist
    if cell_ext[1][1] > img_h:
        cell_ext[1][1] = img_h
    if cell_ext[2][1] > img_h:
        cell_ext[2][1] = img_h

    next_idx, nonzero = calculate_overlap(curr_idx, cell_ext)
    msg = 'move_down: {} -> {}, nonzero {}'.format(curr_idx, next_idx, nonzero)
    logging.info(msg)
    return next_idx

def move_right(curr_idx, cell_list):
    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[2][0] += extend_dist
    cell_ext[3][0] += extend_dist
    if cell_ext[2][0] > img_w:
        cell_ext[2][0] = img_w
    if cell_ext[3][0] > img_w:
        cell_ext[3][0] = img_w

    next_idx, nonzero = calculate_overlap(curr_idx, cell_ext)
    msg = 'move_right: {} -> {}, nonzero {}'.format(curr_idx, next_idx, nonzero)
    logging.info(msg)
    return next_idx

def move_left(curr_idx, cell_list):
    cell_ext = np.vstack([cell_list[curr_idx].x1, cell_list[curr_idx].x2, 
                        cell_list[curr_idx].x3, cell_list[curr_idx].x4])
    cell_ext[0][0] -= extend_dist
    cell_ext[1][0] -= extend_dist
    if cell_ext[0][0] < 0:
        cell_ext[0][0] = 0
    if cell_ext[1][0] < 0:
        cell_ext[1][0] = 0

    next_idx, nonzero = calculate_overlap(curr_idx, cell_ext)
    msg = 'move_left: {} -> {}, nonzero {}'.format(curr_idx, next_idx, nonzero)
    logging.info(msg)
    return next_idx

def gen_grid_map(cell_list):

    layout_map = np.full((map_size,map_size), -1, dtype = np.int8)
    cell_stack = []
    x = map_size //2
    y = map_size //2
    first_cell = 0
    cell_stack.append((first_cell, x, y))

    while cell_stack:
        i, x, y = cell_stack.pop()
        layout_map[y,x] = i

        next_idx = move_down(i, cell_list)
        if next_idx < 0:
            logging.info('stop, move_down, overlap <{}'.format(th_overlap))
        # elif cell_list[next_idx].x1[1] +5 > bottom_left_mark[1]:
            # logging.info('stop, move_down, out of bound, bottom_left')
        elif layout_map[y+1,x]==-1:
            cell_stack.append((next_idx, x, y+1))
            

        next_idx = move_right(i, cell_list)
        if next_idx < 0:
            logging.info('stop, move_right, overlap <{}'.format(th_overlap))
        # elif cell_list[next_idx].x4[0] +5 > top_right_mark[0]:
            # logging.info('stop, move_right, out of bound, top_right')
        elif layout_map[y,x+1]==-1:
            cell_stack.append((next_idx, x+1, y))
           
        next_idx = move_up(i, cell_list)
        if next_idx < 0:
            logging.info('stop, move_up, overlap <{}'.format(th_overlap))
        # elif cell_list[next_idx].x1[1] -5 < top_left_mark[1]:
            # logging.info('stop, move_up, out of bound, top_left')
        elif layout_map[y-1,x]==-1:
            cell_stack.append((next_idx, x, y-1))

        next_idx = move_left(i, cell_list)
        if next_idx < 0:
            logging.info('stop, move_left, overlap <{}'.format(th_overlap))
        # elif cell_list[next_idx].x1[0] -5 < top_left_mark[0]:
            # logging.info('stop, move_left, out of bound, top_left')
        elif layout_map[y,x-1]==-1:
            cell_stack.append((next_idx, x-1, y))

    idx_map = np.argwhere(layout_map>=0)   
    row_idx = idx_map[:,0]
    col_idx = idx_map[:,1]
    x1 = min(col_idx)
    y1 = min(row_idx)
    x2 = max(col_idx)+1
    y2 = max(row_idx)+1

    # print(row_idx, min(row_idx), max(row_idx)) 
    # print(col_idx, min(col_idx), max(col_idx)) 
    # print(idx_map)

    grid_map = layout_map[y1:y2, x1:x2]

    return grid_map


def detect_landmark(small):
    # small = cv2.pyrDown(frame)
    # img_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    img_b, img_g, img_r = cv2.split(small)  

    th, landmark = cv2.threshold(img_b, th_mark, 255, cv2.THRESH_BINARY)
    if args.show_image:
        cv2.imshow('threshold',landmark)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)   

    landmark = cv2.dilate(landmark, landmark_kernel, iterations = 1)

  

    _, contours, hierarchy = cv2.findContours(landmark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    landmark_list = []
    landmark_center_lst = []    
    for con in contours:
        area = cv2.contourArea(con)
        msg = 'detect_landmark: area: {}'.format(area)
        # print(msg)
        logging.info(msg)

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
        msg = 'detect_landmark: ratio: {}'.format(ratio)
        # print(msg)
        logging.info(msg)
        if ratio < mark_ratio_min:
            continue

        M = cv2.moments(con)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # print(cx, cy)
        landmark_center_lst.append([cx,cy])

        # approx = cv2.approxPolyDP(con, 3,True)
        # landmark_list.append(approx)

    landmarks = np.array(landmark_center_lst).reshape(-1, 2)
    logging.info(landmarks)  
    if args.show_image:
        for i in range(landmarks.shape[0]):
            x = landmarks[i, 0]
            y = landmarks[i, 1]
            cv2.circle(landmark,(x, y),6,(54,117,138),-1)

        cv2.imshow('landmark',landmark)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)  
    
    cv2.imwrite('landmark_detection.jpg', landmark)

    return landmarks
     

def detect_grid(small, cell_list):
    global grid_idx_map

    img_b, img_g, img_r = cv2.split(small)  
    img_b = img_b.astype(np.float32)
    img_b = img_b + 1
    ratio_rb = img_r/img_b
    maxrb = np.max(ratio_rb)
    ratio_rb = ratio_rb * 255 / maxrb
    gray = ratio_rb.astype(np.uint8)
    cv2.imwrite('img_rb.jpg', gray)
    # gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,th_block_size,th_c)
    bin_img = cv2.dilate(bin_img, grid_kernel, iterations = 1)
    bin_img = 255 - bin_img
    cv2.imwrite('img_rb_thresh.jpg', bin_img)
    
    _, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if args.show_image:
        bin_color = cv2.cvtColor(bin_img,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(bin_color, contours, -1, (0,255,0), 1)
        cv2.imshow('grid_contour',bin_color)
        key = cv2.waitKey(0)    
    
    center_lst = []    
    for i, con in enumerate(contours):
        area = cv2.contourArea(con)
        if area < cell_area_min or area > cell_area_max:
            msg = 'con {}, area {} not in range ({},{})'.format(i, area, cell_area_min, cell_area_max)
            # print(msg)
            logging.info(msg)              
            continue

        M = cv2.moments(con)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        clen = len(con)
        compact = (clen * clen) / area;
        # print('clen:', clen, compact)
        if compact > cell_compact_th:
            msg = 'con {} [{},{}], compact {} > cell_compact_th {}'.format(i, cx,cy,compact, cell_compact_th)
            # print(msg)
            logging.info(msg)  
            continue

        # rect = cv2.minAreaRect(con)
        # width, height = rect[1]
        # angle = rect[2]
        # ratio = height/width
        # if ratio < cell_ratio_min or ratio > cell_ratio_max:
        #     continue


        # print(cx, cy)
        # center_lst.append([cx,cy])
        approx = cv2.approxPolyDP(con, polygon_dist,True)
        # print('approx nodes:', len(approx), approx)
        if len(approx) not in [4,5]:
            msg = 'con {} [{},{}], X approx nodes: {}'.format(i,cx,cy, len(approx))
            print(msg)
            logging.info(msg)  
            continue

        approx_arr = np.array(approx).reshape(-1, 2)
        # print('approx_arr ',approx_arr)

        cell = Cell([cx,cy], approx_arr)
        cell_list.append(cell)
        
    grid_idx_map = np.full(bin_img.shape, 0, dtype = np.uint8)
    # grid_c3 = cv2.cvtColor(grid_idx_map, cv2.COLOR_GRAY2BGR)
    # cell_list.pop(22)
    for i, cc in enumerate(cell_list):
        cv2.drawContours(grid_idx_map, [cc.polygon], -1, (i+1,i+1,i+1), -1)
    grid_idx_map_1 = grid_idx_map.copy()
    for i, cc in enumerate(cell_list):
        cv2.drawContours(grid_idx_map_1, [cc.polygon], -1, (255,255,255), 1)
    cv2.imwrite('grid_idx_map.png', grid_idx_map_1)

    if args.show_image:
        cv2.imshow('grid_idx_map_1',grid_idx_map_1)
        key = cv2.waitKey(0)    

    msg = 'find {} grid, output to grid_idx_map.png'.format(len(cell_list))
    print(msg)
    logging.info(msg)    

def check_angle(new_rect, landmark_center_lst):
    a = array(landmark_center_lst[new_rect[0]] )
    b = array(landmark_center_lst[new_rect[1]] )
    c = array(landmark_center_lst[new_rect[2]] )
    d = array(landmark_center_lst[new_rect[3]] )

    ret = False
    try :
        u = a - b
        v = c - b
        msg = 'u {}, v {}, dot(u,v) {}, norm(u) {:.2f}, norm(v) {:.2f}'.format(u, v, dot(u,v),norm(u),norm(v))
        logging.info(msg)   
        dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
        angle = arccos(clip(dd, -1, 1))  *180/pi
        # if args.show_debugmsg:
        #     print('angle1:', angle)
        msg = 'a{}, b{}, c{}, angle {:.2f}'.format(a, b, c, angle)
        logging.info(msg)
        if interior_angle_min < angle < interior_angle_max:
            u = b - c
            v = d - c
            dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
            angle = arccos(clip(dd, -1, 1))  *180/pi
            msg = 'b{}, c{}, d{}, angle {:.2f}'.format(b, c, d, angle)
            logging.info(msg)

            if interior_angle_min < angle < interior_angle_max:
                u = c - d
                v = a - d
                dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
                angle = arccos(clip(dd, -1, 1))  *180/pi
                
                msg = 'c{}, d{}, a{}, angle {:.2f}'.format(c, d, a, angle)
                logging.info(msg)

                if interior_angle_min < angle < interior_angle_max:
                    ret = True
    except:
        msg = 'Err: {} check_angle'.format(new_rect)
        print(msg)
        logging.debug(msg)
        ret = True

    if not ret:
        msg = 'check_angle fail min{}, max{}'.format(interior_angle_min,interior_angle_max )
        logging.info(msg)

    return ret

def check_aspect_ratio(new_rect, landmark_center_lst):
    a = array(landmark_center_lst[new_rect[0]] )
    b = array(landmark_center_lst[new_rect[1]] )
    c = array(landmark_center_lst[new_rect[2]] )
    d = array(landmark_center_lst[new_rect[3]] )

    ret = False
    u = a - b
    v = c - d
    du = norm(u)
    dv = norm(v)
    if du < dv:
        ratio1 = du/dv
    else:
        ratio1 = dv/du

    u = b - c
    v = d - a
    du = norm(u)
    dv = norm(v)
    if du < dv:
        ratio2 = du/dv
    else:
        ratio2 = dv/du    

    msg = 'rect ratio: {}, ratio1 {:.2f}, ratio2 {:.2f}'.format(new_rect, ratio1, ratio2)
    logging.info(msg)
    # print(msg)
    if ratio1 > aspect_ratio_th and ratio2 > aspect_ratio_th:
        ret = True

    return ret

def clean_polygon(comb, landmark_center_lst):
    new_comb_lst = []
    num_marks = len(landmark_center_lst)

    for rect in comb: 
        rect_coord = [ landmark_center_lst[i] for i in rect]
        rect_coord = np.array(rect_coord).reshape(-1, 2)
        convex = cv2.convexHull(rect_coord, False)

        msg = 'clean_polygon: rect {}'.format(rect)
        # if args.show_debugmsg:
        #     print(msg)
        #     print(rect_coord)

        logging.info(msg)
        logging.info(rect_coord)

        msg = 'convex {}'.format(convex)
        logging.info(msg)
        # if args.show_debugmsg:
        #     print(msg)        

        if len(convex) < 4:
            msg = 'check convexHull, only 3 points, skip convex {}'.format(rect)
            # print(msg)
            logging.info(msg)
            continue

        inSide = -1
        for i in range(num_marks):
            if i not in rect:
                pts = tuple(landmark_center_lst[i])
                inSideDist = cv2.pointPolygonTest(convex, pts, True )
                if inSideDist > insideDist_th:
                    msg = 'pt {} inside convex {}, skip'.format(i, rect)
                    # print(msg)
                    logging.info(msg)
                    break
        if inSide ==1:
            continue

        approx = cv2.approxPolyDP(convex, polygon_box_dist_th,True)
        if len(approx) <4:
            msg = 'check approxPolyDP, only 3 points, skip convex {}'.format(rect)
            # print(msg)
            logging.info(msg)
            continue            

        convex = np.squeeze(convex)
        new_rect = []
        for pt in convex:
            for i, rectPt in enumerate(rect_coord):
                if np.all(pt==rectPt):
                    new_rect.append(rect[i])
                    break

        logging.info('new_rect:{}'.format(new_rect))
        if check_angle(new_rect, landmark_center_lst):
            if check_aspect_ratio(new_rect, landmark_center_lst):
                new_comb_lst.append(new_rect)
            else:
                msg = 'check_aspect_ratio failed, skip convex {}'.format(new_rect)
                # print(msg)
                logging.info(msg)                  
        else:
            msg = 'check_angle failed, skip convex {}'.format(new_rect)
            # print(msg)
            logging.info(msg)           
  
    return new_comb_lst

def find_rect_combination(landmark_center_lst):
    mark_lst = [i for i in range(len(landmark_center_lst))]
    comb = list(combinations(mark_lst, 4))
    comb = clean_polygon(comb, landmark_center_lst)
    
    msg = 'clean_polygon \n {}'.format(comb)
    logging.info(msg) 
    # print(msg)

    num_comb = len(comb)
    maxarea = 0
    minarea = np.iinfo(np.int32).max
    found = False

    for i in range(num_comb-1): 
        for j in range(i+1, num_comb):
            intersect = set(comb[i]) & set(comb[j])
            if len(intersect) !=1: # and len(intersect) !=2:
                msg = '{} {} {} interset test failed'.format(comb[i], comb[j], intersect)
                logging.info(msg)
                # print(msg)
                continue

            coord1 = [landmark_center_lst[i] for i in comb[i]]    
            poly1 = Polygon(coord1)
            coord2 = [landmark_center_lst[i] for i in comb[j]] 
            poly2 = Polygon(coord2)

            intersect = poly1.intersection(poly2) 

            msg = 'find_rect_combination {}, {}, area {:.1f}'.format(comb[i], comb[j], intersect.area)
            logging.info(msg) 
            print(msg)  

            if intersect.area < 0.1:
                area = poly1.area+ poly2.area
                if area > maxarea:
                    maxarea = area
                    p1 = i
                    p2 = j
                    found = True
            else:
                if intersect.area < minarea:
                    minarea = intersect.area
                    q1 = i
                    q2 = j
    
    if found:
        msg = 'find_rect_combination, intersect 0 {}, {}'.format(comb[p1], comb[p2])
        logging.info(msg) 
        print(msg)   
        rect_lst = [comb[p1], comb[p2]]
    else:
        msg ='find_rect_combination, find min intersect area {}, {} {:.1f}'.format(
            comb[q1], comb[q2], minarea)
        logging.debug(msg) 
        print(msg)     
        rect_lst = [comb[q1], comb[q2]]      
    # print(poly1.area, poly2.area, poly1.area+ poly2.area)

    return rect_lst



def get_rect_map(grid_map, rect, landmarks, cell_list, small_grid_img):
    vertex_lst = [ landmarks[i] for i in rect]
    rect_vertex = np.array(vertex_lst).reshape(-1, 2)
    msg = 'get_rect_map: rect: {}\n{}'.format(rect, rect_vertex)
    logging.info(msg)
    print(msg)

    landmark_coord = find_top_left_mark(rect_vertex)  
    rect_coord = np.array(landmark_coord).reshape(-1, 2)
    cv2.drawContours(small_grid_img, [rect_coord], -1, (0, 0, 0), 2)

    mark_cell_idx = [-1, -1, -1, -1]
    for k, m in enumerate(landmark_coord):
        min_dist = 999999
        for i, cell in enumerate(cell_list):
            if k==0:
                dist = np.linalg.norm(np.array(m - cell.x1))
            elif k==1:
                 dist = np.linalg.norm(np.array(m - cell.x2))
            elif k==2:
                dist = np.linalg.norm(np.array(m - cell.x3))
            elif k==3:
                dist = np.linalg.norm(np.array(m - cell.x4))

            if dist < min_dist:
                min_dist = dist
                idx = i

        if min_dist < left_top_dist_gap:
            mark_cell_idx[k] = idx

    msg = 'mark_cell_idx {}'.format(mark_cell_idx)
    print(msg)
    logging.info(msg)

    mark_cell_coord = []
    for i in range(4):
        if mark_cell_idx[i] != -1:
            coord = np.argwhere(grid_map ==mark_cell_idx[i])
            msg = 'find mark_cell_idx {}, coord {} '.format(mark_cell_idx[i], coord)
            print(msg)
            logging.info(msg)
            mark_cell_coord.append([mark_cell_idx[i], coord[0,0], coord[0,1]])
        else:
            mark_cell_coord.append([mark_cell_idx[i], -1, -1])

    print('before fill:',mark_cell_coord)
    logging.info('before fill: {}'.format(mark_cell_coord))
    if mark_cell_coord[0][0]==-1 :
        mark_cell_coord[0][1] = mark_cell_coord[3][1]  # row
        mark_cell_coord[0][2] = mark_cell_coord[1][2]  # col
    elif mark_cell_coord[1][0]==-1:
        mark_cell_coord[1][1] = mark_cell_coord[2][1]  # row
        mark_cell_coord[1][2] = mark_cell_coord[0][2]  # col
    elif mark_cell_coord[2][0]==-1 :
        mark_cell_coord[2][1] = mark_cell_coord[1][1]  # row
        mark_cell_coord[2][2] = mark_cell_coord[3][2]  # col
    elif mark_cell_coord[2][0]==-1 :
        mark_cell_coord[3][1] = mark_cell_coord[0][1]  # row
        mark_cell_coord[3][2] = mark_cell_coord[2][2]  # col

    print('after fill:',mark_cell_coord)
    logging.info('after fill: {}'.format(mark_cell_coord))

    no_mark_cell = False
    for i in range(4):
        if mark_cell_coord[i][1]==-1:
            no_mark_cell = True
            logging.debug('mark_cell_coord {} = -1'.format(i))
            break

    if no_mark_cell:
        return grid_map
    else:
        mask = np.full(grid_map.shape, True) # dtype=np.uint8)
        row1 = mark_cell_coord[0][1]
        row2 = mark_cell_coord[1][1]
        col1 = mark_cell_coord[0][2]
        col2 = mark_cell_coord[3][2]

        mask[row1:row2+1, col1:col2+1] = False

        rect_map = grid_map.copy()
        rect_map[mask] = -1
        logging.info('rect_map:\n{}'.format(rect_map))
        return rect_map


def prune_grid_map(grid_map, landmarks, cell_list, small_grid_img):
    num_landmarks = len(landmarks)
    if num_landmarks > 4:
        rect_lst = find_rect_combination(landmarks)    
        if len(rect_lst) != 2:
            msg = 'Error: number of rect should be 2, not {}, {}'.format(
                len(rect_lst), rect_lst)
            print(msg)
            logging.debug(msg)
            return None

        print(rect_lst)

        rect_map1 = get_rect_map(grid_map, rect_lst[0], landmarks, cell_list, small_grid_img)
        rect_map2 = get_rect_map(grid_map, rect_lst[1], landmarks, cell_list, small_grid_img)
        mask1 = rect_map1 <0
        mask2 = rect_map2 <0
        mask = (mask1 & mask2)
        grid_map[mask] = -1
        # print(mask )

    else:
        rect_lst = [0,1,2,3]
        rect_map1 = get_rect_map(grid_map, rect_lst, landmarks, cell_list, small_grid_img)
        mask1 = rect_map1 <0
        grid_map[mask1] = -1

    return grid_map


def identify_grid(landmark_img, grid_img):
    global img_w
    global img_h

    cell_list = []
    center_lst = []

    small_grid_img = cv2.pyrDown(grid_img)
    small_landmark = cv2.pyrDown(landmark_img)
    img_w = small_landmark.shape[1]
    img_h = small_landmark.shape[0]
    print('identify_grid: small image size (w, h): ', img_w, img_h)

    landmarks = detect_landmark(small_landmark)
    num_landmarks = len(landmarks)
    for i in range(landmarks.shape[0]):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        cv2.circle(small_grid_img,(x, y),5,(54,117,138),-1)

    # cv2.imshow('grid_detection circle',small_grid_img)
    # key = cv2.waitKey(0)

    msg = 'num_landmarks: {}'.format(num_landmarks)
    logging.info(msg)
    print(msg)
    if  num_landmarks !=4 and num_landmarks !=7:
        msg = 'Error: num_landmarks should be 4 or 7, not {}'.format(num_landmarks)
        print(msg)
        logging.debug(msg)
        return None


    detect_grid(small_grid_img, cell_list)

    for i, cc in enumerate(cell_list):
        cv2.drawContours(small_grid_img, [cc.polygon], -1, (0,255,0), 1)
        strText = '{}'.format(i)
        cv2.putText(small_grid_img, strText, (cc.center[0], cc.center[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.circle(small_grid_img,(cc.center[0], cc.center[1]),2,(0,0,255),2)
        center_lst.append(cc.center)
    # cv2.drawContours(small, landmark_list, -1, (0,0, 255), 1)

    for i, cc in enumerate(landmarks):
        strText = '[{}]'.format(i)
        cv2.putText(small_grid_img, strText, (cc[0], cc[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255,0))
        cv2.circle(small_grid_img,(cc[0], cc[1]),2,(0,255,255),2)

    grid_map = gen_grid_map(cell_list)
    print(grid_map)
    logging.info(grid_map)

    cv2.imwrite('grid_detection0.jpg', small_grid_img)
    if args.show_image:
        cv2.imshow('grid_detection0',small_grid_img)
        key = cv2.waitKey(0)

    grid_map = prune_grid_map(grid_map, landmarks, cell_list, small_grid_img)

    final_grid_map = np.full((map_size_out,map_size_out), -1, dtype = np.int8)
    final_grid_map[:grid_map.shape[0],:grid_map.shape[1]] = grid_map[:,:]

    logging.info('prune_grid_map : \n{}'.format(grid_map))
    logging.info('final_grid_map : \n{}'.format(final_grid_map))

    print('prune_grid_map : \n',grid_map)
    print('final_grid_map : \n',final_grid_map)

    cv2.imwrite('grid_detection1.jpg', small_grid_img)
    if args.show_image:
        cv2.imshow('grid_detection1',small_grid_img)
        key = cv2.waitKey(0)



    return final_grid_map, center_lst

def main():
    global ini_disable_picam
    global ini_show_debugmsg
    global ini_show_image
    global args  

    parser = argparse.ArgumentParser(description='grid')   
    parser.add_argument('--disable_picam', action="store_true", dest='disable_picam', default=ini_disable_picam, help='disable picam')
    parser.add_argument('--enable_picam', action="store_false", dest='disable_picam', default=ini_disable_picam, help='enable picam')
    parser.add_argument('--show_image', action="store_true", dest='show_image', default=ini_show_image, help='show debug image')
    parser.add_argument('--noshow_image', action="store_false", dest='show_image', default=ini_show_image, help='no show debug image')
    parser.add_argument('--show_debugmsg', action="store_true", dest='show_debugmsg', default=ini_show_debugmsg, help='show debug message')
    parser.add_argument('--noshow_debugmsg', action="store_false", dest='show_debugmsg', default=ini_show_debugmsg, help='no show debug message')
    parser.add_argument('-fm', action="store", dest='landmark_name', default='landmark.jpg', help='input landmark image name')
    parser.add_argument('-fg', action="store", dest='grid_name', default='grid.jpg', help='input grid image name')    
    args = parser.parse_args()

    print('disable_picam:', args.disable_picam)
    print('show_image:   ', args.show_image)
    print('show_debugmsg:', args.show_debugmsg)    

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


    if args.disable_picam:
        print('load landmark image:', args.landmark_name)
        landmark_img = cv2.imread(args.landmark_name)
        
        print('load grid image:', args.grid_name)
        grid_img = cv2.imread(args.grid_name)
        
    else:
        camera = PiCamera()
        camera.awb_mode = 'off'
        camera.awb_gains = (1.7,1.7)
        camera.exposure_mode =  'off'

        camera.iso = 100
        camera.resolution = (1296,976)

        camera.framerate = 5
        camera.shutter_speed = shutter_speed_landmark
        input("按 <ENTER> 開始拍定位燈號影像 ...")
        camera.capture('landmark.jpg')
        print('output landmark.jpg')

        camera.shutter_speed = shutter_speed_grid
        input("按 <ENTER> 開始拍淨空格子影像...")
        camera.capture('grid.jpg')
        print('output grid.jpg')

        landmark_img = cv2.imread('landmark.jpg') #'3000.jpg'
        grid_img = cv2.imread('grid.jpg') #'50000.jpg


    if landmark_img is None:
        print('cannot read landmark.jpg')
        logging.debug("cannot read landmark.jpg")
    elif grid_img is None:
        print('cannot read grid.jpg')
        logging.debug("cannot read grid.jpg")
    else:
        grid_map, center_lst =identify_grid(landmark_img, grid_img)

        f = open('grid_map.pickle', 'wb')
        pickle.dump(grid_map, f)
        f.close()

        f = open('center_lst.pickle', 'wb')
        pickle.dump(center_lst, f)
        f.close()

        print('output file: grid_map.pickle')        
        print('output file: center_lst.pickle')  
        print('output file: img_rb.jpg') 
        print('output file: img_rb_thresh.jpg') 
        print('output file: grid_idx_map.png') 
        print('output file: grid_detection0.jpg') 
        print('output file: grid_detection1.jpg')
        print('output file: landmark_detection.jpg')

if __name__ == "__main__":
    main()