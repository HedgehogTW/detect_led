# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:52:19 2018

@author: cclee
"""
import argparse
import cv2
import numpy as np
from numpy import linalg as LA

from datetime import datetime
import scipy.stats
import os, sys    
import time  
from time import localtime, strftime 
import logging
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2": # linux
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    ini_disable_picam = False
    ini_disable_mqtt = False
    ini_show_debugmsg = False
    ini_show_image = False
    wait_time_ms = 10
else: # PC, read video
    ini_disable_picam = True
    ini_disable_mqtt = True
    ini_show_debugmsg = True
    ini_show_image = True
    wait_time_ms = 0

args = None


bright_th = 60 #50
blob_low = 35
blob_up = 700
cc_area_low = 10
cc_area_up = 50
nonzero_frame = blob_low # 50 for cc, 150 for blob
cc_dist = 30

fps = 0
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
# old_blob_frame = None
led_on = False
blob_lst = []


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    logging.info("MQTT: Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(mqtt_topic) #$SYS/#")

def on_disconnect(client, userdata,rc=0):
    print("DisConnected result code "+str(rc))
    logging.info("MQTT: DisConnected result code "+str(rc))
    client.loop_stop()

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print('\ton_message:', msg.topic+" "+str(msg.payload))
    logging.info('\tMQTT: on_message:'+ msg.topic+" "+str(msg.payload))
	
def on_publish(client, userdata, mid):
    print('\ton_publish:', mid)
    logging.info('\tMQTT: on_publish:{}'.format(mid))


def find_mode(data, bw):
    # try:
    #     kde_r = scipy.stats.gaussian_kde(data, bw_method=bw) 
    #     maxP = 0
    #     mode = 0
    #     for i in range(256):
    #         p = kde_r.evaluate(i)
    #         # print(i, p)
    #         if p >= maxP:
    #             maxP = p
    #             mode = i  
    # except:
    #     mode = np.median(data)

    mode = np.median(data)
    return mode

def detect_blob_color(frame, blob_lst, mask_led):

    if args.disable_picam:
        img_b, img_g, img_r = cv2.split(frame)  
    else:
        img_r, img_g, img_b = cv2.split(frame)  
    # color_lst = []
    # bounding_lst = []

    for i, bbdict in enumerate(blob_lst):
        box = bbdict['box'] 
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        mask_zero = np.zeros(mask_led.shape, dtype="uint8") 
        cv2.rectangle(mask_zero,(x,y),(x+w,y+h),(255,255,255),-1)

        mask = mask_led & mask_zero
        pts_red = img_r[mask==255]
        # print('len(pts_red)',len(pts_red))
        mode_r = find_mode(pts_red, 3)

        pts_green = img_g[mask==255]
        mode_g = find_mode(pts_green, 3)
        
        pts_blue = img_b[mask==255]
        mode_b = find_mode(pts_blue, 3)

        if args.show_debugmsg:
            print('R G B:', mode_r, mode_g, mode_b)

        if mode_r >210 and mode_g > 210 and mode_b > 210:
            bbdict['color'] = 'white'
        elif mode_r > mode_g and mode_r > mode_b:
            bbdict['color'] = 'red'
            # color_lst.append('red')
        elif mode_b > mode_g and mode_b > mode_r:
            bbdict['color'] = 'blue'
            # color_lst.append('blue')
        else: #if mode_g > mode_b and mode_g > mode_r:
            bbdict['color'] = 'green'
            # color_lst.append('green')

    # remove while color blob    
    blob_lst = [bb for bb in blob_lst if bb['color'] != 'white'] 
    return blob_lst

# def track_blobs(img_bin_now):
#     overlap = True
#     # if old_blob_frame is not None:
#     for bbdict in blob_lst:
#         box = bbdict['box'] 
#         x = box[0]
#         y = box[1]
#         w = box[2]
#         h = box[3]
#         rect = img_bin_now[y:y+h, x:x+w]
#         # cv2.rectangle(img_bin_now,(x,y),(x+w,y+h),(255,255,255),1)
#         # cv2.imshow("old_bin_frame", img_bin_now)
#         count = np.count_nonzero(rect)
#         if count < cc_area_low:
#             bbdict['on'] = False
#             overlap = False
#             # print(bbdict['color'] + 'off')

#             # print(box, count)
#     return overlap


def find_blobs(frame):
    global led_on
    global blob_lst
    # global old_blob_frame

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, img_bin = cv2.threshold(img_gray, bright_th, 255, cv2.THRESH_BINARY)
    # img_bin_op = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel3)
    img_bin_now = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel13)
    count = np.count_nonzero(img_bin_now)

    if count < blob_low:
        led_on = False
        blob_lst = []
        old_blob_frame = None
        return 0

    # if led_on:
    #     overlap = track_blobs(img_bin_now)
    #     if overlap:
    #         # old_blob_frame = img_bin_now.copy()
    #         print('led_on overlap ---------------')
    #     else:
    #         print('no overlap ---------------')
    #         for blob in blob_lst:
    #             if blob['on'] ==False:
    #                 print(blob['color'] + ' off')



    blob_lst = []
    # cv2.imshow("img_bin_now", img_bin_now)
    _, contours, hierarchy = cv2.findContours(img_bin_now, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for con in contours:
        area = cv2.contourArea(con)
        x,y,w,h = cv2.boundingRect(con)
        
        if args.show_image:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),1)

        logging.info('blob area {}, boundingRect (x{},y{},w{},h{})'.format(area,x,y,w,h))
        if args.show_debugmsg:
            print('blob area ', area, ', boundingRect (x,y,w,h):', x,y,w,h)

        if blob_low < area < blob_up:
            bbdict = {'con':None, 'cen':None, 'color':None,'box':None, 'on':True, 'grid':None, 'area':0}

            
            bbdict['area'] = area
            bbdict['con'] = con
            bbdict['box'] = (x,y,w,h)
            blob_lst.append(bbdict)

    for blob in blob_lst:
        con = blob['con']
        conarr = con.reshape(con.shape[0], 2)
        center = np.mean(conarr, axis = 0, dtype=np.float32)
        # cen_lst.append((center[0], center[1]))
        blob['cen']= (center[0], center[1])


    blobs = len(blob_lst)
    if blobs > 0:
        # color_lst, bounding_lst = detect_blob_color(frame, blob_lst, img_bin_op)
        detect_blob_color(frame, blob_lst, img_bin_now)
        # old_blob_frame = img_bin_now.copy()
        led_on = True
    else:
        led_on = False
        # old_blob_frame = None

    return blobs
    
def locate_position():
    global blob_lst   
    for j, blob in enumerate(blob_lst):
        center = blob['cen']
        r = int(center[1])
        c = int(center[0])
        str_position = '{}{}'.format(chr(65+int(r/ ygap)), int(c/ xgap))
        blob['grid'] = str_position
        
        
        # payload = "LED ON {}--({},{}), {}".format(blob['color'], c, r, str_position)      
        # client.publish(mqtt_topic, payload, qos)
        # print(payload)

def process_frame(frame):
    t1 = datetime.now()
    numBlobs = find_blobs(frame)
    locate_position()

    if args.show_debugmsg:
        print('find_blobs:', numBlobs)  
    logging.info('find_blobs:{}'.format(numBlobs))

    for j, blob in enumerate(blob_lst):       
        payload = "LED ON {}--{} {}".format(blob['color'], blob['grid'], blob['area'])   
        logging.info(payload) 
        if not args.disable_mqtt:  
            logging.info('sending mqtt ...')            
            if args.show_debugmsg:            
                print('sending mqtt ...')
            client.publish(mqtt_topic, payload, qos)

        if args.show_debugmsg:
            print(payload)

        # print('{}, {}, {}, {}'.format(frame_num, j, blob['area'], blob['color']), file=outfile)
        # logging.info('{}, {}, {}, {}'.format(frame_num, j, blob['area'], blob['color']))

    t2 = datetime.now()
    delta = t2 - t1
    if args.show_debugmsg:
        print('--Frame:{} Computation time takes {}'.format(frame_num, delta))

    terminate = False
    if args.show_image:
        for i, bbdict in enumerate(blob_lst):
            box = bbdict['box'] 
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.putText(frame, bbdict['color'], (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
        # cv2.imwrite('many_result.jpg',frame)

    
        cv2.imshow("frame", frame)
        key = cv2.waitKey(wait_time_ms)
        if key == 27:
            cv2.destroyAllWindows()
            # break
            terminate = True

    return terminate

def process_picam():
    global frame_num
    global fps
    global xgap
    global ygap
    
    
    with PiCamera() as camera:
        camera.awb_mode = 'off'
        camera.awb_gains = (1.7,1.7)
        camera.exposure_mode =  'off'
        camera.iso = 100
        camera.resolution = (1296,972)
        camera.zoom = (0,0.5,0.5,0.5)
        camera.framerate = 5
        camera.shutter_speed = 1000
        if args.enable_record:
            camera.start_recording('/home/pi/detect_led/picam_rec.h264')
        # camera.annotate_text = "isoAWBoffEV-1"      

        fps = camera.framerate

        print('awb_gains {}, iso {} shutter_speed {}'.format(camera.awb_gains, camera.iso, camera.shutter_speed))

        with PiRGBArray(camera) as rawCapture:
            camera.capture(rawCapture, format="bgr")
            width = rawCapture.array.shape[1]
            height = rawCapture.array.shape[0]
            xgap = width / num_grid 
            ygap = height / num_grid 

            print('width {}, height {} fps {}'.format(width, height, fps))
            logging.info('width {}, height {} fps {}'.format(width, height, fps))

            rawCapture.truncate(0)    
            # outfile = open('area.csv', 'w')
            while True:
                # try:
                frame_num += 1
                # if frame_num % sampling_rate:
                    # continue

                camera.capture(rawCapture, format="bgr") #ambilgambar

                # if args.show_debugmsg:
                #     print('Captured %dx%d image, (%02d:%02d:%02d.%d)' % (
                #         rawCapture.array.shape[1], rawCapture.array.shape[0],
                #         t1.hour, t1.minute, t1.second, t1.microsecond/1000))
                

                frame = rawCapture.array
                terminate = process_frame(frame)
                if terminate:
                    break

                # cv2.imwrite('frame_%04d.jpg'%frame_num, frame)
                if args.show_image:
                    cv2.imshow('image',frame)
                    # cv2.imwrite('test.jpg', frame)
                rawCapture.truncate(0)
                # except:
                #     logging.debug("Except: Something else went wrong")
                #     print("Except: Something else went wrong") 

                    
                time.sleep(0.5)

            if args.enable_record:
                camera.stop_recording()

            # outfile.close()
        # camera.stop_preview()

def process_video_file(filename):
    global frame_num
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
        
    # outfile = open('area.csv', 'w')
    while True:

        bVideoRead, frame = cap.read()  
        # print(frame.shape)
        if bVideoRead==False:
            break
        frame_num += 1

        if frame_num % sampling_rate:
            continue
        # cv2.imwrite('test.jpg',frame)
        # frame = cv2.imread('many.jpg')

        terminate = process_frame(frame)
        if terminate:
            break

    # outfile.close()

   
def clean_oldlog_files(logpath):
    now = datetime.now()
    files = os.listdir(logpath)
    for f in files:
        try:
            ftitle = os.path.splitext(f)[0]
            log_date = datetime.strptime(ftitle, '%Y-%m-%d-%H%M%S')
            delta = now - log_date
            if delta.days > 30:
                fullpath = os.path.join(logpath, f)
                os.remove(fullpath)
                if args.show_debugmsg:
                    print('remove old log:', f)
        except:
            print('ERROR: clean_oldlog_files:', f)
            continue


def main():
    global client
    global ini_disable_mqtt
    global ini_disable_picam
    global ini_show_image
    global ini_show_debugmsg
    global args    

    parser = argparse.ArgumentParser(description='detect led')   
    parser.add_argument('--disable_mqtt', action="store_true", dest='disable_mqtt', default=ini_disable_mqtt, help='disable mqtt')
    parser.add_argument('--enable_mqtt', action="store_false", dest='disable_mqtt', default=ini_disable_mqtt, help='enable mqtt')
    parser.add_argument('--disable_picam', action="store_true", dest='disable_picam', default=ini_disable_picam, help='disable picam')
    parser.add_argument('--enable_picam', action="store_false", dest='disable_picam', default=ini_disable_picam, help='enable picam')
    parser.add_argument('--enable_record', action="store_true", dest='enable_record', default=False, help='enable picam record')
    parser.add_argument('--show_image', action="store_true", dest='show_image', default=ini_show_image, help='show debug image')
    parser.add_argument('--noshow_image', action="store_false", dest='show_image', default=ini_show_image, help='no show debug image')
    parser.add_argument('--show_debugmsg', action="store_true", dest='show_debugmsg', default=ini_show_debugmsg, help='show debug message')
    parser.add_argument('--noshow_debugmsg', action="store_false", dest='show_debugmsg', default=ini_show_debugmsg, help='no show debug message')
    parser.add_argument('-f', action="store", dest='video_name', default='2_(1).h264', help='input video file name')
    args = parser.parse_args()
    
    print('disable_mqtt: ', args.disable_mqtt)
    print('disable_picam:', args.disable_picam)
    print('enable_record:', args.enable_record)
    print('show_image:   ', args.show_image)
    print('show_debugmsg:', args.show_debugmsg)                

    logpath = os.path.dirname('log/') 
    if not os.path.exists(logpath):
        os.makedirs(logpath)
        print('no output log path, create one') 
    else:
        clean_oldlog_files(logpath)


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
    logging.info("Start of the program")


    if not args.disable_mqtt:
        print('mqtt qos:', qos)
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        
        client.connect("140.138.178.116", 1883, 60)
        client.loop_start()
        client.publish(mqtt_topic, 'video start', qos)
    

    if args.disable_picam:
        print('process video file:', args.video_name)
        process_video_file(args.video_name)
    else:
        process_picam()
    
    if not args.disable_mqtt:
        client.publish(mqtt_topic, 'video end', qos)
        time.sleep(1)
        client.disconnect()



if __name__ == "__main__":
    main()