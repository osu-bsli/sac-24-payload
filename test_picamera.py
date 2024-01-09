#!/usr/bin/python3
import subprocess
import numpy as np
import cv2
import os
import time
import sys
import math
import glob
import signal

# clear ram
pics = glob.glob('/run/shm/test*.jpg')
for t in range(0,len(pics)):
    os.remove(pics[t])
def Camera_start(wx,hx):
    global p
    rpistr = "libcamera-vid -t 10 --segment 1 --codec mjpeg -n -o /run/shm/test%06d.jpg --width " + str(wx) + " --height " + str(hx)
    p = subprocess.Popen(rpistr, shell=True, preexec_fn=os.setsid)
#initialise variables
width        = 720
height       = 540
start = 0
cv2.namedWindow('Frame')
Text = "Left Mouse click on picture to EXIT"
ttrat = time.time()
#define mouse clicks (LEFT to EXIT, RIGHT to switch eye detcetion ON/OFF)
def mouse_action(event, x, y, flags, param):
    global p
    if event == cv2.EVENT_LBUTTONDOWN:
        os.killpg(p.pid, signal.SIGTERM)
        cv2.destroyAllWindows()
        sys.exit()
   
cv2.setMouseCallback('Frame',mouse_action)
# start capturing images
Camera_start(width,height)
# main loop
while True:
    # remove message after 3 seconds
    if time.time() - ttrat > 3 and ttrat > 0:
        Text =""
        ttrat = 0
    # load image
    pics = glob.glob('/run/shm/test*.jpg')
    while len(pics) < 2:
        pics = glob.glob('/run/shm/test*.jpg')
    pics.sort(reverse=True)
    img = cv2.imread(pics[1])
    if len(pics) > 2:
        for tt in range(2,len(pics)):
            os.remove(pics[tt])
    
    cv2.imshow('Frame',img)
    cv2.waitKey(10)
