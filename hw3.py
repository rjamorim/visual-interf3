# Visual Interfaces Spring 2015 Assignment 3
# Roberto Amorim - rja2139

import cv2, cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

image = cv2.imread("ass3-labeled.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
#image = cv2.imread("ass3-campus.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
contours = cv2.findContours(image.copy(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_NONE)

# Here we open the building names table
try:
    with open("ass3-table.txt", 'rb') as f:
        buildings = f.readlines()
except IOError:
    print "ERROR: The file containing building names can not be read"
    exit()
f.close()
names = []
names.append("null")
for line in buildings:
    line = line.rstrip('\r\n')
    toks = line.split('=')
    names.append(toks[1].replace('"', ''))

shapes = []
for array in contours[0]:
    # Here we get the color (index):
    color = image[array[0][0][1]][array[0][0][0]]

    # Here we find the area of each shape
    area = cv2.contourArea(array)

    # And here the center of mass
    moments = cv2.moments(array)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
    center = (cx, cy)

    cv2.circle(image, center, 3, 128, 1)


# print shapes

#cv2.imshow('campus', image)
#cv2.waitKey(0)