# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:52:19 2018

@author: cclee
"""

import cv2
import numpy as np
from datetime import datetime
from sklearn.neighbors import KernelDensity
import sys, getopt        
#from matplotlib import pyplot as plt
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep
import io

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
class Blob:
    def __init__(self, c, h):
        self.coord = c;
        self.hue = h;
        self.counter = 0;
        self.willRemove = False;
        self.bOn = False;
        self.id = ++blob_id;
        self.color = ''
        self.position = ''
#        self.detect_color();    

    def detect_color(self):
        diff_cyan = abs(self.hue-180)
        diff_red = abs(self.hue - 360)
        diff_red1 = abs(self.hue)
        if diff_cyan < diff_red and diff_cyan < diff_red1:
            self.color = "blue";
        else:
            self.color = "red";
            

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
    
def find_blobs(img_bin):
    con_candidate = []
    areaLower = 50
    areaUpper = 1500
    _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for con in contours:
        area = cv2.contourArea(con)
        if areaLower < area < areaUpper:
            con_candidate.append(con)
#            print(con)            
    return con_candidate

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

def get_hue(hue, contours, idx):
    mask = np.zeros(hue.shape, dtype="uint8") 
    cv2.drawContours(mask, contours, idx, 255, -1) 
#    plt.imshow(mask, 'gray')
#    plt.imshow(mask, 'hue')
#    print(mask.shape, hue.shape, hue.dtype)
    hue1 = hue[mask==255]
    hue2 = hue1.reshape(-1, 1)
#    print(hue1.shape, len(hue2))
#    hist = cv2.calcHist([hue],[0],mask,[180],[0,180])    
#    kde = stats.gaussian_kde(hue1, bw_method=17)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(hue2)
    maxkde = 0
    maxh = 0
    for h in range(180):
        log_dens = kde.score_samples(h)
        p = np.exp(log_dens)
#        print(p, end=' ')
        if p >= maxkde:
#            maxkde = kde(h)
            maxkde = p
            maxh = h
            
    return maxh*2

    
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

    with PiCamera() as camera:
        with PiRGBArray(camera) as rawCapture:
            #camera.awb_mode = 'off'
            camera.exposure_compensation = (-15)
            camera.iso = 100
            camera.resolution = (640,480)
            camera.framerate = 5
            # rawCapture = PiRGBArray(camera)
            # camera.start_preview()
            #for effect in camera.IMAGE_EFFECTS:
                #camera.image_effect = effect
            #camera.image_effect = ''
            camera.start_recording('/home/pi/1.h264')
            camera.annotate_text = "isoAWBoffEV-1"
     
            fps = camera.framerate
            width = 640
            height = 480
            xgap = width / num_grid ;
            ygap = height / num_grid ;
                
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            while True:
                t1 = datetime.now()
                # bVideoRead, frame = cap.read()  
                # if bVideoRead==False:
                    # break
                frame_num += 1

                if frame_num % sampling_rate:
                    continue

                
                camera.capture(rawCapture, format="bgr") #ambilgambar
                print('Captured %dx%d image, (%02d:%02d:%02d.%d)' % (
                    rawCapture.array.shape[1], rawCapture.array.shape[0],
                    t1.hour, t1.minute, t1.second, t1.microsecond/1000))
                

                frame = rawCapture.array

                cv2.imshow('image',frame)
                cv2.imwrite('test.jpg', frame)
                rawCapture.truncate(0)

                # break
                

        #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #         img_h, img_s, img_v = cv2.split(hsv)  
        #         th, img_bin = cv2.threshold(img_v,240, 255,cv2.THRESH_BINARY)
        #         img_bin = cv2.medianBlur(img_bin,7)
        #         img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
        #         contours = find_blobs(img_bin)
        #         numBlobs = len(contours)
        # #//	cv::imshow("binary", bin);

        #         if numBlobs <= 0: 
        #             for b in lstBlob:
        #                 b.willRemove = True
        #             remove_blobs()
        #         else:
        #             identify_blobs(contours, img_h)

                t2 = datetime.now()
                delta = t2 - t1
                # print('    Computation time takes {}'.format(delta))

                # sleep(20)

                key = cv2.waitKey(10)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key == 32:
                    cv2.destroyAllWindows()
                    break 
         
            camera.stop_recording()
        # camera.stop_preview()

def main():
    global client
    print('len(sys.argv):', len(sys.argv))
    print('mqtt qos:', qos)
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
#    client.connect("140.138.178.116", 1883, 60)
#    client.loop_start()
    
    process_video('video640-mkv.mkv');
    
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