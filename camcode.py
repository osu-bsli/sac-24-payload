#!/usr/bin/env python
import time
#import board
import os
#from picamera import PiCamera
import cv2
# from PIL import Image
#from intertools import product

#print(f'CV2 version is {cv2.__version__}')
camera = cv2.VideoCapture()
timestart = time.time()
os.system("libcamera-vid -t 1000 --code mjpeg --segment 1 -o cam/full%05d.jpeg")
#os.system("libcamera-vid -t 10000 --codec yuv420 -o test.data")
#while (time.time() - timestart < 5):
#	os.system("libcamera-still -e png -o cam/test.png")
# result, img = camera.read()
timetook = time.time() - timestart
print("Time it took is " +str(timetook) + " seconds")
# print(result, img)






#note :images are 640x480

#splitting the images test
'''
filename = "full00000.jpeg"
dir_in = "cam/"
dir_out = "cam/"
d = 240
def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
'''
