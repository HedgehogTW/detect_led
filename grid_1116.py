import cv2
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from math import pi
# from numpy import linalg as LA
import argparse
import logging
import os, sys  
import time  
from time import localtime, strftime 
from itertools import combinations 
from shapely.geometry import Polygon
# import pathlib

# 底下是 detect_grid 參數
grid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
landmark_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
th_block_size = 33
th_c = -20
cell_area_min = 1500
cell_area_max = 10000
cell_ratio_min = 0.15
cell_ratio_max = 0.55
cell_compact_th = 50
polygon_dist = 10


# 底下是 detect_landmark 參數
th_mark = 240
mark_area_min = 40
mark_area_max = 150
mark_ratio_min = 0.75

# 底下是 find_layout 參數
extend_dist = 30
th_overlap = 100
img_w = None
img_h = None

# rect_combination
polygon_box_dist = 20
map_size = 15

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


def find_top_left_mark(landmark_center_lst, logging):
    # global top_left_mark
    # global bottom_left_mark
    # global bottom_right_mark
    # global top_right_mark

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

def move_down(curr_idx, cell_list, logging, debug=False):
    cell_mask = np.zeros(grid_idx_map.shape, dtype=np.uint8)
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

    msg = 'move_down: idx {}, nonzero {}'.format(next_idx, nonzero)
    # if debug:
    #     print(msg)
    logging.info(msg)
    return next_idx

def move_right(curr_idx, cell_list, logging, debug=False):
    cell_mask = np.zeros(grid_idx_map.shape, dtype=np.uint8)
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

    msg = 'move_right: idx {}, nonzero {}'.format(next_idx, nonzero)
    # if debug:
    #     print(msg)
    logging.info(msg)
    return next_idx

def layout_cells(landmark_coord, cell_list, logging, debug=False):
    # global layout_map
    layout_map = np.full((map_size,map_size), -1, dtype = np.int8)

    top_left_mark = landmark_coord[0]
    bottom_left_mark = landmark_coord[1]
    top_right_mark = landmark_coord[3]

    min_dist = 999999    
    for i, cell in enumerate(cell_list):
        # print(i, cell.top_left)
        dx = cell.x1[0]-top_left_mark[0]
        dy = cell.x1[1]-top_left_mark[1] 

        dist = dx**2 + dy **2
        if dist < min_dist:
            min_dist = dist
            first_cell = i

    if debug:
        print('first cell ', first_cell)
    logging.info('first cell {}'.format(first_cell))

    # cell_mask = np.zeros(grid_idx_map.shape, dtype=int)
    layout_map[0, 0] = first_cell
    col = 0
    row = 1
    num_rows = -1
    all_Done = False
    while True:
        i = first_cell

        while True: # find next row
            next_idx = move_down(i, cell_list, logging, debug)
            if next_idx < 0:
                # if debug:
                #     print('break, find_layout move_down count_nonzero <', th_overlap)

                logging.info('break, find_layout move_down count_nonzero <{}'.format(th_overlap))
                break

            if num_rows ==-1 :
                if cell_list[next_idx].x1[1] +5 > bottom_left_mark[1]:
                    msg = 'bottom_left_mark bound, break'
                    # if debug:
                    #     print(msg)
                    logging.info(msg)
                    break
            elif row>=  num_rows:
                msg = '> num_rows, break'
                # if debug:
                #     print(msg)
                logging.info(msg)
                break 

     
            # print(next_idx)
            layout_map[row,col] = next_idx
            cell_list[i].set_coord((row,col))
            i = next_idx
            row += 1   

        if col==0:
            num_rows = row
            print('num_rows ', num_rows)

        row = 0
        while True: # find next column
            next_col = move_right(first_cell, cell_list, logging, debug)
            if next_col < 0:
                msg = 'continue, {} move_right {} count_nonzero < {}'.format(first_cell,next_col,th_overlap)
                # if debug:
                #     print(msg)

                logging.info(msg)
                row += 1
                first_cell = layout_map[row,col] 
                continue
            elif cell_list[next_col].x4[0] -10 > top_right_mark[0]:
                msg = 'continue, {} move_right {} next_col x4 x {} > top_right_mark x {}'.format(
                    first_cell,next_col, cell_list[next_col].x4[0], top_right_mark[0])
                # if debug:
                #     print(msg)
                logging.info(msg)

                row += 1  
                first_cell = layout_map[row,col]   
                           
                continue
            elif row >=  num_rows:
                all_Done = True
                break 
            
            break

        if all_Done:
            break

        col += 1
        first_cell = next_col
        layout_map[row,col] = next_col
        cell_list[next_col].set_coord((row,col))
        row += 1 

    col += 1
    return layout_map, row, col


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

    landmark_centers = np.array(landmark_center_lst).reshape(-1, 2)
    # print(landmark_centers)
    return landmark_centers
     

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
        # print('approx nodes:', len(approx), approx)
        if len(approx) != 4:
            print('X approx nodes:', len(approx))
            continue

        approx_arr = np.array(approx).reshape(-1, 2)
        # print('approx_arr ',approx_arr)

        cell = Cell([cx,cy], approx_arr)
        cell_list.append(cell)
        
    grid_idx_map = np.full(bin_img.shape, 0, dtype = np.uint8)
    # grid_c3 = cv2.cvtColor(grid_idx_map, cv2.COLOR_GRAY2BGR)
    for i, cc in enumerate(cell_list):
        cv2.drawContours(grid_idx_map, [cc.polygon], -1, (i+1,i+1,i+1), -1)
    cv2.imwrite('grid_idx_map.png', grid_idx_map)

def check_angle(new_rect, landmark_center_lst):
    a = array(landmark_center_lst[new_rect[0]] )
    b = array(landmark_center_lst[new_rect[1]] )
    c = array(landmark_center_lst[new_rect[2]] )
    d = array(landmark_center_lst[new_rect[3]] )

    ret = False
    try :
        u = a - b
        v = c - b
        dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
        angle = arccos(clip(dd, -1, 1))  *180/pi
        if 60 < angle < 120:
            u = b - c
            v = d - c
            dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
            angle = arccos(clip(dd, -1, 1))  *180/pi
            if 60 < angle < 120:
                u = c - d
                v = a - d
                dd = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
                angle = arccos(clip(dd, -1, 1))  *180/pi
                if 60 < angle < 120:
                    ret = True
    except:
        msg = 'Err: {} check_angle'.format(new_rect)
        print(msg)
        logging.debug(msg)
        ret = True

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
    # print(ratio1, ratio2)
    if ratio1 > 0.4 and ratio2 > 0.4:
        ret = True

    return ret

def clean_polygon(comb, landmark_center_lst, logging):
    new_comb_lst = []
    num_marks = len(landmark_center_lst)

    for rect in comb: 
        rect_coord = [ landmark_center_lst[i] for i in rect]
        rect_coord = np.array(rect_coord).reshape(-1, 2)
        # print(rect)
        logging.info(rect)
        convex = cv2.convexHull(rect_coord, False)
        if len(convex) < 4:
            msg = 'check convexHull, only 3 points, skip convex {}'.format(rect)
            print(msg)
            logging.info(msg)
            continue

        inSide = -1
        for i in range(num_marks):
            if i not in rect:
                pts = tuple(landmark_center_lst[i])
                inSide = cv2.pointPolygonTest(convex, pts, False )
                if inSide ==1:
                    msg = '{} inside convex {}, skip'.format(i, rect)
                    print(msg)
                    logging.info(msg)
                    break
        if inSide ==1:
            continue

        approx = cv2.approxPolyDP(convex, polygon_box_dist,True)
        if len(approx) <4:
            msg = 'check approxPolyDP, only 3 points, skip convex {}'.format(rect)
            print(msg)
            logging.info(msg)
            continue            

        convex = np.squeeze(convex)
        new_rect = []
        for pt in convex:
            i = np.argwhere(rect_coord ==pt)[0][0]
            new_rect.append(rect[i])

        if check_angle(new_rect, landmark_center_lst):
            if check_aspect_ratio(new_rect, landmark_center_lst):
                new_comb_lst.append(new_rect)
            else:
                msg = 'check_aspect_ratio failed, skip convex {}'.format(new_rect)
                print(msg)
                logging.info(msg)                  
        else:
            msg = 'check_angle failed, skip convex {}'.format(new_rect)
            print(msg)
            logging.info(msg)           
  
    return new_comb_lst

def find_rect_combination(landmark_center_lst, logging):
    mark_lst = [i for i in range(len(landmark_center_lst))]
    comb = list(combinations(mark_lst, 4))
    comb = clean_polygon(comb, landmark_center_lst, logging)
    
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
            if len(intersect) !=1 and len(intersect) !=2:
                msg = '{} {} {} interset test failed'.format(comb[i], comb[j], intersect)
                logging.info(msg)
                print(msg)
                continue


            coord1 = [landmark_center_lst[i] for i in comb[i]]    
            poly1 = Polygon(coord1)
            coord2 = [landmark_center_lst[i] for i in comb[j]] 
            poly2 = Polygon(coord2)

            intersect = poly1.intersection(poly2) 

            msg = 'find_rect_combination {}, {}'.format(comb[i], comb[j])
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

def layout_cell_2rect(rect_lst, landmark_centers, cell_list, logging, debug=False):

############# rect 0
    vertex_lst = [ landmark_centers[i] for i in rect_lst[0]]
    rect_vertex = np.array(vertex_lst).reshape(-1, 2)
    msg = 'layout_cell_2rect: rect_0: {}\n{}'.format(rect_lst[0], rect_vertex)
    logging.info(msg)
    print(msg)

    landmark_coord1 = find_top_left_mark(rect_vertex, logging)  
    layout_map1, row1, col1 = layout_cells(landmark_coord1, cell_list, logging, debug)
    logging.info(layout_map1)
    print(layout_map1, row1, col1)

############# rect 1
    vertex_lst = [ landmark_centers[i] for i in rect_lst[1]]
    rect_vertex = np.array(vertex_lst).reshape(-1, 2)
    msg = 'layout_cell_2rect: rect_1: {}\n{}'.format(rect_lst[1], rect_vertex)
    logging.info(msg)
    print(msg)

    landmark_coord2 = find_top_left_mark(rect_vertex, logging)  
    layout_map2, row2, col2 = layout_cells(landmark_coord2, cell_list, logging, debug)
    logging.info(layout_map2)
    print(layout_map2, row2, col2)

############# intersect
    intersect = list(set(rect_lst[0]) & set(rect_lst[1]))
    msg = 'layout_cell_2rect: intersect {} '.format(intersect)
    logging.info(msg)
    print(msg)

    if len(intersect) !=1 :
        msg = 'Error: layout_cell_2rect: {} & {}= {}, intersection point >1'.format(rect_lst[0], rect_lst[1], intersect)
        logging.debug(msg)
        print(msg)
        layout_map = layout_map2
    else:
        inter_pt = landmark_centers[intersect[0]]

        if inter_pt in landmark_coord1[0]:
            logging.info('intersection pt on the top left of rect0')
            print('intersection pt on the top left of rect0')
            layout_map2[row2:row2+row1, 0:col1] = layout_map1[0:row1, 0:col1]  
            layout_map = layout_map2

        elif inter_pt in landmark_coord1[1]:
            logging.info('intersection pt on the bottom left of rect0')
            print('intersection pt on the bottom left of rect0')
            layout_map1[row1:row1+row2, 0:col2] = layout_map2[0:row2, 0:col2]  
            layout_map = layout_map1

        elif inter_pt in landmark_coord1[2]:
            logging.info('intersection pt on the bottom right of rect0')
            print('intersection pt on the bottom right of rect0')
            if row1 >= row2:
                layout_map1[row1-row2:row1, col1:col1+col2] = layout_map2[0:row2, 0:col2]  
                layout_map = layout_map1
            else:
                layout_map = np.full((map_size,map_size), -1, dtype = np.int8)
                layout_map[row2-row1:row2, 0:col1] = layout_map1[0:row1, 0:col1] 
                layout_map[0:row2, col1:col1+col2] = layout_map2[0:row2, 0:col2] 

        elif inter_pt in landmark_coord1[3]:
            logging.info('intersection pt on the top right of rect0')
            print('intersection pt on the top right of rect0')
            layout_map1[0:row2, col1:col1+col2] = layout_map2[0:row2, 0:col2]  
            layout_map = layout_map1        


    return layout_map

def identify_grid(frame, logging, debug=False):
    global img_w
    global img_h

    cell_list = []
    center_lst = []

    small = cv2.pyrDown(frame)
    img_w = small.shape[1]
    img_h = small.shape[0]
    print('identify_grid: small image size (w, h): ', img_w, img_h)

    detect_grid(small, cell_list, logging, debug)

    landmark_centers = detect_landmark(small, logging, debug)
    num_landmarks = len(landmark_centers)
    msg = 'num_landmarks: {}'.format(num_landmarks)
    logging.info(msg)
    print(msg)
    if  num_landmarks !=4 and num_landmarks !=7:
        msg = 'Error: num_landmarks should be 4 or 7, not {}'.format(num_landmarks)
        print(msg)
        logging.debug(msg)
        return None

    if num_landmarks > 4:
        rect_lst = find_rect_combination(landmark_centers, logging)
        if len(rect_lst) != 2:
            msg = 'Error: number of rect combinations should be 2, not {}, {}'.format(
                len(rect_lst), rect_lst)
            print(msg)
            logging.debug(msg)
            return None

        print(rect_lst)

        layout_map = layout_cell_2rect(rect_lst, landmark_centers, cell_list, logging, debug)



    else:
        landmark_coord = find_top_left_mark(landmark_centers, logging)  
        layout_map, row, col = layout_cells(landmark_coord, cell_list, logging, debug)

    logging.info(layout_map)
    print(layout_map)

    for i, cc in enumerate(cell_list):
        cv2.drawContours(small, [cc.polygon], -1, (0,255,0), 1)
        strText = '{}'.format(i)
        cv2.putText(small, strText, (cc.center[0], cc.center[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        cv2.circle(small,(cc.center[0], cc.center[1]),2,(0,0,255),2)
        center_lst.append(cc.center)
    # cv2.drawContours(small, landmark_list, -1, (0,0, 255), 1)

    for i, cc in enumerate(landmark_centers):
        strText = '[{}]'.format(i)
        cv2.putText(small, strText, (cc[0], cc[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0, 200))

    cv2.imwrite('grid_detection.png', small)

    if debug:
        cv2.imshow('cell_list',small)
        key = cv2.waitKey(0)

    return layout_map, center_lst

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
        # cv2.imwrite('grid_img.png', frame)
        frame = cv2.imread('grid_img3.jpg')
        
        if frame is None:
            print('cannot read grid_img.png')
        else:
            identify_grid(frame, logging, args.show_debugmsg)



