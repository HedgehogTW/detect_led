from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.awb_mode = 'off'
camera.awb_gains = (1.7,1.7)
camera.exposure_mode =  'off'

camera.iso = 100
camera.resolution = (1296,976)

camera.framerate = 5
camera.shutter_speed = 5000

camera.capture('test.jpg')

