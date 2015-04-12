# Visual Interfaces Spring 2015 Assignment 3
# Roberto Amorim - rja2139

import cv2, cv
import numpy as np

charact = []

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
# We create an array "names" that can associate building names with their index colors
names = []
names.append("null")
for line in buildings:
    line = line.rstrip('\r\n')
    toks = line.split('=')
    names.append(toks[1].replace('"', ''))


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


# Here we detect which shape is the squarest
def squarest(shapes):
    most = 10.0
    building = 0
    for shape in shapes:
        # We only care about buildings that approach a rectangle
        if shape[4] != 0:
            ma = float(max(shape[1][2], shape[1][3]))
            mi = float(min(shape[1][2], shape[1][3]))
            if ma / mi < most:
                most = ma / mi
                building = shape[0]
    return building


# Here we detect which shape is the squarest
def mostrectangular(shapes):
    most = 0.0
    building = 0
    for shape in shapes:
        # We only care about buildings that approach a rectangle
        if shape[4] != 0:
            ma = float(max(shape[1][2], shape[1][3]))
            mi = float(min(shape[1][2], shape[1][3]))
            if ma / mi > most:
                most = ma / mi
                building = shape[0]
    return building


# Here we detect which shape is the largest
def largest(shapes):
    most = float("-inf")
    building = 0
    for shape in shapes:
        if shape[2] > most:
            most = shape[2]
            building = shape[0]
    return building


# Here we detect which shape is the smallest
def smallest(shapes):
    most = float("inf")
    building = 0
    for shape in shapes:
        if shape[2] < most:
            most = shape[2]
            building = shape[0]
    return building


# Here we detect which shape is the longest
def longest(shapes):
    most = 0
    building = 0
    for shape in shapes:
        if shape[1][2] > most:
            most = shape[1][2]
            building = shape[0]
        if shape[1][3] > most:
            most = shape[1][3]
            building = shape[0]
    return building


# Here we detect which shape is the thinnest
def thinnest(shapes):
    most = float("inf")
    building = 0
    for shape in shapes:
        if shape[1][2] < most:
            most = shape[1][2]
            building = shape[0]
        if shape[1][3] < most:
            most = shape[1][3]
            building = shape[0]
    return building


# Here we cluster areas to decide which shapes are "small", "medium" and "large"
def areaclust(shapes):
    # Start by creating an array with areas. To suit kmeans' finnicky rules, we must reshape the array into
    # a matrix with a single column and convert the values to float32
    areas = []
    for shape in shapes:
        areas.append(shape[2])
    areas = np.array(areas)
    areas = areas.reshape((27, 1))
    areas = np.float32(areas)
    # cluster criteria: either 10 max iteractions or epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # calculate the clusters: 3 of them (small, medium and large), starting with random centers
    compactness, labels, centers = cv2.kmeans(areas, 3, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    i = 0
    for label in labels:
        if label == 0:
            charact.append([shapes[i][0], "small building"])
        elif label == 1:
            charact.append([shapes[i][0], "medium building"])
        elif label == 2:
            charact.append([shapes[i][0], "large building"])
        i += 1


# Here we detect whether the shape is "quadrilateral-ish", that is, approaches a rectangle;
# or with substantial negative space
def quadrilateral(mbr, shape):
    flag = 0
    area_shape = cv2.contourArea(shape)
    area_mbr = mbr[2] * mbr[3]
    if ( area_mbr / area_shape ) < 1.26:
        flag = 1
    return flag


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

    quadr = quadrilateral((x, y, w, h), array)

    # Here we find the area of each shape
    area = int(cv2.contourArea(array))

    # And here the center of mass
    moments = cv2.moments(array)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
    center = (cx, cy)

    shapes.append([color, (x, y, w, h), area, center, quadr])

    #cv2.circle(image, center, 3, 128, 1)

charact.append([northernmost(shapes), "northernmost building"])
charact.append([southernmost(shapes), "southernmost building"])
charact.append([westernmost(shapes), "westernmost building"])
charact.append([easternmost(shapes), "easternmost building"])
charact.append([squarest(shapes), "squarest building"])
charact.append([mostrectangular(shapes), "most rectangular building"])
charact.append([largest(shapes), "largest building"])
charact.append([smallest(shapes), "smallest building"])
charact.append([longest(shapes), "longest building"])
charact.append([thinnest(shapes), "thinnest building"])
areaclust(shapes)

for character in charact:
    print names[character[0]] + ": " + character[1]

#cv2.imshow('campus', image)
#cv2.waitKey(0)