import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
th_block_size = 33
th_c = -20
cell_area_min = 1200
cell_area_max = 10000
cell_ratio_min = 0.15
cell_ratio_max = 0.55
cell_compact_th = 50
polygon_dist = 10
nBins = 12
# cell_map = np.full((20, 6), -1, dtype=int)

def detect_grid(frame, debug=False):
    small = cv2.pyrDown(frame)
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    # lower_line = np.array([12,0,128])
    # upper_line = np.array([45,191,255])
    # mask = cv2.inRange(hsv,lower_line,upper_line)
    # res = cv2.bitwise_and(frame,frame,mask=mask)


    bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,th_block_size,th_c)
    bin_img = cv2.dilate(bin_img, kernel, iterations = 1)
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

    # print(histo)
    # print(bin_edges)

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




    # for ll in col_edges:
    #     bx1 = ll[0]
    #     bx2 = ll[1]
    #     bc = ll[2]
    #     cv2.line(small, (bx1, 0), (bx1, small.shape[0]), (0, 0, 255), 2)
    #     cv2.line(small, (bx2, 0), (bx2, small.shape[0]), (0, 0, 255), 2)
    #     cv2.line(small, (bc, 0), (bc, small.shape[0]), (0, 255, 255), 2)


    # print('find {} cells'.format(len(cell_list)))

    cv2.drawContours(small, cell_list, -1, (0,255,0), 1)
    if debug:
        cv2.imshow('cell_list',small)
        # cv2.imshow('bin_color',bin_color)
        key = cv2.waitKey(0)


    cv2.imwrite('grid_detection.png', small)


    return coord_dict, cell_list, center_lst

if __name__ == "__main__":
    frame = cv2.imread('grid_img.png')
    detect_grid(frame)