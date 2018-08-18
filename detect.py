# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:52:19 2018

@author: cclee
"""

import cv2
import numpy as np
from numpy import linalg as LA
from datetime import datetime
import scipy.stats
# from scipy import stats

import sys, getopt        
#from matplotlib import pyplot as plt
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

bright_th = 30 #50
blob_low = 60
blob_up = 400
cc_area_low = 10
cc_area_up = 50
nonzero_frame = blob_low # 50 for cc, 150 for blob
cc_dist = 30

fps = 0
BLOB_DIST = 20
blob_id = 0 
lstBlob = []
frame_num = 0
num_grid = 8
xgap = 0
ygap = 0
mqtt_topic = 'cclee/led'
client = None
qos = 0
loop_timeout = 2
sampling_rate = 4

kernel13 = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
old_blob_frame = None
led_on = False

cen_lst = []
blob_lst = []
color_lst = []
bounding_lst = []
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(mqtt_topic) #$SYS/#")

def on_disconnect(client, userdata,rc=0):
    print("DisConnected result code "+str(rc))
    client.loop_stop()

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print('\ton_message:', msg.topic+" "+str(msg.payload))
	
def on_publish(client, userdata, mid):
    print('\ton_publish:', mid)
    


def find_mode(data, bw):
    kde_r = scipy.stats.gaussian_kde(data, bw_method=bw) 
    maxP = 0
    mode = 0
    for i in range(256):
        p = kde_r.evaluate(i)
        # print(i, p)
        if p >= maxP:
            maxP = p
            mode = i  

    return mode

def detect_blob_color(frame, blob_lst, mask_led):

    img_b, img_g, img_r = cv2.split(frame)  
    color_lst = []
    bounding_lst = []

    for i, cnt in enumerate(blob_lst):
        mask_zero = np.zeros(mask_led.shape, dtype="uint8") 

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(mask_zero,(x,y),(x+w,y+h),(255,255,255),-1)
        bounding_lst.append((x,y,w,h))

        mask = mask_led & mask_zero
        pts_red = img_r[mask==255]
        print('len(pts_red)',len(pts_red))
        mode_r = find_mode(pts_red, 3)

        pts_green = img_g[mask==255]
        mode_g = find_mode(pts_green, 3)
        
        pts_blue = img_b[mask==255]
        mode_b = find_mode(pts_blue, 3)
        print(mode_r, mode_g, mode_b)

        if mode_r > mode_g and mode_r > mode_b:
            color_lst.append('red')
        elif mode_b > mode_g and mode_b > mode_r:
            color_lst.append('blue')
        else: #if mode_g > mode_b and mode_g > mode_r:
            color_lst.append('green')

    return color_lst, bounding_lst

# def check_overlap(img_bin_ok):
#     global cen_lst
#     global old_blob_frame

#     numBlobs = len(cen_lst)
#     overlap = img_bin_ok & old_blob_frame
#     _, contours, hierarchy = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     overlap_blobs = len(contours)

#     if numBlobs ==overlap_blobs:
#         return True
#     else:
#         return False

def track_blobs():
    if old_blob_frame is not None:
        for b in bounding_lst:
            x = b[0]
            y = b[1]
            w = b[2]
            h = b[3]
            rect = old_blob_frame[y:y+h, x:x+w]
            cv2.rectangle(old_blob_frame,(x,y),(x+w,y+h),(255,255,255),1)
            cv2.imshow("old_bin_frame", old_blob_frame)
            count = np.count_nonzero(rect)
            print(b, count)



def find_blobs(frame):
    global led_on
    global cen_lst
    global blob_lst
    global color_lst
    global bounding_lst
    global old_blob_frame


    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, img_bin = cv2.threshold(img_gray, bright_th, 255, cv2.THRESH_BINARY)
    img_bin_op = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel3)
    img_bin_ok = cv2.morphologyEx(img_bin_op, cv2.MORPH_CLOSE, kernel13)
    count = np.count_nonzero(img_bin_ok)

    if count > blob_low:
        if led_on and old_blob_frame is not None:
            track_blobs(img_bin_ok)
            overlap = check_overlap(img_bin_ok)
            if overlap:
                old_blob_frame = img_bin_ok.copy()
                print('led_on overlap ---------------')
                return


        cen_lst = []
        blob_lst = []
        color_lst = []
        bounding_lst = []

        _, contours, hierarchy = cv2.findContours(img_bin_ok, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for con in contours:
            area = cv2.contourArea(con)
            print('area ', area)
            if blob_low < area < blob_up:
                bbdict = {'con':None, 'cen':None, 'color':None,'box':None}
                bbdict['con'] = con
                blob_lst.append(con)

        for con in blob_lst:
            conarr = con.reshape(con.shape[0], 2)
            center = np.mean(conarr, axis = 0, dtype=np.float32)
            cen_lst.append((center[0], center[1]))

        blobs = len(blob_lst)
        if blobs > 0:
            color_lst, bounding_lst = detect_blob_color(frame, blob_lst, img_bin_op)
            old_blob_frame = img_bin_ok.copy()
            led_on = True
        else:
            led_on = False
            old_blob_frame = None

    else:
        led_on = False
        cen_lst = []
        blob_lst = []
        color_lst = []
        bounding_lst = []
        old_blob_frame = None

    # return blob_lst, cen_lst, color_lst, bounding_lst 

def remove_blobs():
    global lstBlob
    global client
#    sz = len(lstBlob)
    i = 0
    while i < len(lstBlob):
        b = lstBlob[i]
        if b.willRemove:
            if b.bOn:
                payload = 'LED OFF hue {}--{}, {}'.format(b.hue, b.color, b.position) 

                client.publish(mqtt_topic, payload, qos)
#                client.loop(loop_timeout)
                print(payload)
#            print('remove {}'.format(b.id))
            lstBlob.pop(i)
            i-=1
        i += 1

    
def identify_blobs(contours, img_h):
    global lstBlob
    global fps
    global client
    
    for b in lstBlob:
        b.willRemove = True
        
    for j, con in enumerate(contours):
        coord = [0,0]
        h = 0
        for pt in con:
            coord += pt
            h += img_h[pt[0, 1], pt[0, 0]]*2
        coord = coord / len(con)        
        h = get_hue(img_h, contours, j);	
        
        bFound = False       
        for b in lstBlob:
            diff_pt = coord - b.coord
            dist = cv2.norm(diff_pt)
            if dist < BLOB_DIST:
                b.hue = h
                b.coord = coord
                b.counter += 1
                b.willRemove = False
                b.detect_color()
                r = int(coord[0, 1])
                c = int(coord[0, 0])
                b.position = '{}{}'.format(chr(65+int(r/ ygap)), int(c/ xgap))
                bFound = True
                
                if b.counter==fps//sampling_rate:
                    b.bOn = True
#                    print('coord ',coord, h, frame_num)
                    payload = "LED ON hue {}--{}, {}".format(h, b.color, b.position)
                    client.publish(mqtt_topic, payload, qos)
#                    client.loop(loop_timeout)
                    print(payload)
                break
            
        if bFound==False:
            blob = Blob(coord, h)
            lstBlob.append(blob)
#            print('add ', blob.id)
            
def process_video(filename):
    global frame_num
    global lstBlob
    global fps
    global xgap
    global ygap
    
    print('process video:', filename)
    cap = cv2.VideoCapture(filename)
    bOpenVideo = cap.isOpened()
    print('Open Video: {0} '.format(bOpenVideo))
    if bOpenVideo == False:
        return  
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width {}, height {} fps {}'.format(width, height, fps))

    xgap = width / num_grid ;
    ygap = height / num_grid ;
        

    while True:
        t1 = datetime.now()
        bVideoRead, frame = cap.read()  
        # print(frame.shape)
        if bVideoRead==False:
            break
        frame_num += 1

        if frame_num % sampling_rate:
            continue
        # cv2.imwrite('test.jpg',frame)
        # frame = cv2.imread('many.jpg')

        
        # blob_lst, cen_lst, color_lst, bounding_lst  = find_blobs(frame)
        find_blobs(frame)
        numBlobs = len(cen_lst)


        print('find_blobs:', numBlobs, cen_lst)  
        print('color_lst:', color_lst)

        t2 = datetime.now()
        delta = t2 - t1
        print('{} Computation time takes {}'.format(frame_num, delta))

        for i, b in enumerate(bounding_lst):
            x = b[0]
            y = b[1]
            w = b[2]
            h = b[3]            
            cv2.putText(frame, color_lst[i], (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
        # cv2.imwrite('many_result.jpg',frame)

    
        cv2.imshow("frame", frame)
        # cv2.imshow("red blob binary", mask_led)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
#        elif key == 32:
#            cv2.destroyAllWindows()
#            break 


        # if numBlobs <= 0: 
        #     for b in lstBlob:
        #         b.willRemove = True
        #     remove_blobs()
        # else:
        #     identify_blobs(contours, img_h)


            
def main():
    global client
    print('len(sys.argv):', len(sys.argv))
    print('mqtt qos:', qos)
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
#    client.connect("140.138.178.116", 1883, 60)
#    client.loop_start()
    
    process_video('2 (2).h264');
    
#    client.publish(mqtt_topic, 'video end', qos)
    
#    client.disconnect()


#    
#    try:
#        opts, args = getopt.getopt(sys.argv[1:], "1234")
#    except getopt.GetoptError as err:
#        # print help information and exit:
#        print( str(err))
#        print('detect.py -1 or detect.py -2')             
#        return 2
#
#    for o, a in opts:
#        if o == "-1":
#            print('process one_lightvideo640-mkv.mkv ...')
#            process_video('one_lightvideo640-mkv.mkv');
#        elif o == '-2':
#            print('process video640-mkv.mkv...')
#            process_video('video640-mkv.mkv')
#        else:
#            return 0
        
if __name__ == "__main__":
    main()