# Visual Interfaces Spring 2015 Assignment 3
# Roberto Amorim - rja2139

import cv2, cv
import numpy as np

image = cv2.imread("ass3-labeled.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
#image = cv2.imread("ass3-campus.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
contours = cv2.findContours(image.copy(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_NONE)


# Here we detect which shape is the northernmost
def northernmost(shapes):
    most = float("inf")
    building = 0
    for shape in shapes:
        if shape[1][1] < most:
            most = shape[1][1]
            building = shape[0]
    return building


# Here we detect which shape is the southernmost
def southernmost(shapes):
    most = float("-inf")
    building = 0
    for shape in shapes:
        if (shape[1][1]+shape[1][3]) > most:
            most = shape[1][1]+shape[1][3]
            building = shape[0]
    return building


# Here we detect which shape is the westernmost
def westernmost(shapes):
    most = float("inf")
    building = 0
    for shape in shapes:
        if shape[1][0] < most:
            most = shape[1][0]
            building = shape[0]
    return building


# Here we detect which shape is the easternmost
def easternmost(shapes):
    most = float("-inf")
    building = 0
    for shape in shapes:
        if shape[1][0]+shape[1][2] > most:
            most = shape[1][0]+shape[1][2]
            building = shape[0]
    return building


# Here we detect


# Here we open the building names table
try:
    with open("ass3-table.txt", 'rb') as f:
        buildings = f.readlines()
except IOError:
    print "ERROR: The file containing building names can not be read"
    exit()
f.close()
# We create an array "names" that can associate building names with their index colors
names = []
names.append("null")
for line in buildings:
    line = line.rstrip('\r\n')
    toks = line.split('=')
    names.append(toks[1].replace('"', ''))

shapes = []
# Here we analyze each shape individually, obtaining basic information such as area and MBR
for array in contours[0]:
    # Here we get the color (index):
    color = image[array[0][0][1]][array[0][0][0]]

    # Here we obtain the minimum bounding rectangles
    x, y, w, h = cv2.boundingRect(array)

    #print names[color]
    #print (float(max(w, h)) / float(min(w, h)))
    #print w
    #print h

    # Here we find the area of each shape
    area = int(cv2.contourArea(array))

    # And here the center of mass
    moments = cv2.moments(array)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
    center = (cx, cy)

    shapes.append([color, (x, y, w, h), area, center])

    # cv2.circle(image, center, 3, 128, 1)

descriptions = []
squarest = 0
mostrectangular = 0
smallest = 0
largest = 0

print names[northernmost(shapes)]
print names[southernmost(shapes)]
print names[westernmost(shapes)]
print names[easternmost(shapes)]


#cv2.imshow('campus', image)
#cv2.waitKey(0)