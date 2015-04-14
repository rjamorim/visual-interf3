# Visual Interfaces Spring 2015 Assignment 3
# Roberto Amorim - rja2139

import cv2, cv
import numpy as np

# Array that holds characteristics for the campus buildings
charact = []

image = cv2.imread("ass3-labeled.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
#image = cv2.imread("ass3-campus.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
contours = cv2.findContours(image.copy(), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_NONE)

# Here we open the building names table
try:
    with open("ass3-table.txt", 'rb') as f:
        buildings = f.readlines()
    f.close()
except IOError:
    print "ERROR: The file containing building names can not be read"
    exit()
# We create an array "names" that can associate building names with their index colors
names = []
names.append("None")
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
            # The squarest building is the one with smallest ratio height/width
            if ma / mi < most:
                most = ma / mi
                building = shape[0]
    return building


# Here we detect which shape is the most rectangular
def mostrectangular(shapes):
    most = 0.0
    building = 0
    for shape in shapes:
        # We only care about buildings that approach a rectangle
        if shape[4] != 0:
            ma = float(max(shape[1][2], shape[1][3]))
            mi = float(min(shape[1][2], shape[1][3]))
            # The most rectangular building is the one with largest ratio height/width
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


# Here we decide whether a building is I-shaped or C-shaped
def lettershape(shape):
    # Method that returns the amount of sides in the shape polygon
    poly = cv2.approxPolyDP(shape, 0.009*cv2.arcLength(shape, True), True)
    hull = cv2.convexHull(shape, returnPoints=False)
    defects = cv2.convexityDefects(shape, hull)
    point = 0
    try:
        for point in range(defects.shape[0]):
            pass
        point += 1
    except:
        point = 0
    # If the shape has 12 sides and two convexity points, it looks like an I
    if point == 2 and len(poly) == 12:
        return "I-shaped building"
    # If the shape has eight sides and one convexity point, it looks like a C
    if point == 1 and len(poly) == 8:
        return "C-shaped building"
    else:
        return None


# Here we decide whether a building has "chewed" corners
def corners(shape):
    # Method that returns the amount of sides in the shape polygon
    poly = cv2.approxPolyDP(shape, 0.009*cv2.arcLength(shape, True), True)
    hull = cv2.convexHull(shape, returnPoints=False)
    defects = cv2.convexityDefects(shape, hull)
    point = 0
    try:
        for point in range(defects.shape[0]):
            pass
        point += 1
    except:
        point = 0
    # If the shape has 12 or 16 sides and 4 convexity points, it has "chewed corners"
    if point == 4 and (len(poly) == 12 or len(poly) == 16):
        return "'Chewed' corners building"
    else:
        return None


# Figures the geographical position of the building within the campus
def geoposition(c):
    # First we divide the map dimensions in thirds
    width = len(image)
    height = len(image[0])
    ythird = width / 3
    xthird = height / 3
    if c[0] < xthird and c[1] < ythird:
        return "Building position: northwest"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] < ythird:
        return "Building position: north"
    elif c[0] > xthird*2 and c[1] < ythird:
        return "Building position: northeast"
    elif c[0] < xthird and c[1] > ythird and c[1] < ythird*2:
        return "Building position: west"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] > ythird and c[1] < ythird*2:
        return "Building position: center"
    elif c[0] > xthird*2 and c[1] > ythird and c[1] < ythird*2:
        return "Building position: east"
    elif c[0] < xthird and c[1] > ythird*2:
        return "Building position: southwest"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] > ythird*2:
        return "Building position: south"
    elif c[0] > xthird*2 and c[1] < ythird*2:
        return "Building position: southeast"
    else:
        return None


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
# or if it contains substantial negative space in the mbr
def quadrilateral(mbr, shape):
    flag = 0
    area_shape = cv2.contourArea(shape)
    area_mbr = mbr[2] * mbr[3]
    # I decided on values of 1.25 and lower for the areas ratio as a good approximation for a rectangular shape
    if ( area_mbr / area_shape ) < 1.26:
        flag = 1
    return flag


shapes = []
# Here we analyze each shape individually, obtaining basic information such as area and MBR
for shape in contours[0]:
    # Here we get the color (index):
    color = image[shape[0][0][1]][shape[0][0][0]]

    # Here we obtain the minimum bounding rectangle
    x, y, w, h = cv2.boundingRect(shape)

    quadr = quadrilateral((x, y, w, h), shape)

    # Here we find the area of the shape
    area = int(cv2.contourArea(shape))

    # And here the center of mass
    moments = cv2.moments(shape)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
    center = (cx, cy)

    letter = lettershape(shape)
    if letter:
        charact.append([color, letter])
        charact.append([color, "Sharp corners building"])
    corn = corners(shape)
    if corn:
        charact.append([color, corn])

    position = geoposition(center)
    if position:
        charact.append([color, position])

    # All values are added to an array that will be used elsewhere in the program
    shapes.append([color, (x, y, w, h), area, center, quadr])


    #cv2.circle(image, center, 3, 128, -1)

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