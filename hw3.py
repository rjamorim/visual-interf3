# Visual Interfaces Spring 2015 Assignment 3
# Roberto Amorim - rja2139

import cv2, cv
import time, math
import numpy as np

# Array that holds characteristics for the campus buildings
charact = []

image = cv2.imread("ass3-labeled.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
display = cv2.imread("ass3-campus.pgm", cv.CV_LOAD_IMAGE_UNCHANGED)
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
        if shape[4]:
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
        if shape[4]:
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
    # Method that calculates the convex hull of the shape
    hull = cv2.convexHull(shape, returnPoints=False)
    # Method that identifies each convexity defect in the hull
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
        return "'chewed' corners building"
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
        return "building position: northwest"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] < ythird:
        return "building position: north"
    elif c[0] > xthird*2 and c[1] < ythird:
        return "building position: northeast"
    elif c[0] < xthird and c[1] > ythird and c[1] < ythird*2:
        return "building position: west"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] > ythird and c[1] < ythird*2:
        return "building position: center"
    elif c[0] > xthird*2 and c[1] > ythird and c[1] < ythird*2:
        return "building position: east"
    elif c[0] < xthird and c[1] > ythird*2:
        return "building position: southwest"
    elif c[0] > xthird and c[0] < xthird*2 and c[1] > ythird*2:
        return "building position: south"
    elif c[0] > xthird*2 and c[1] < ythird*2:
        return "building position: southeast"
    else:
        return None


# Here we decide whether the shape is oriented north-south or east-west
def orientation(dimen):
    l = float(max(dimen[0], dimen[1]))
    s = float(min(dimen[0], dimen[1]))
    # I reckon that if the reason between the largest dimension and the smallest dimension is smaller than
    # 1.25, that shape is too close to a square for the orientation to be visually meaningful
    if l / s < 1.25:
        return None
    else:
        if dimen[0] > dimen[1]:
            return "oriented east-west (horizontal)"
        else:
            return "oriented north-south (vertical)"


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


# Here we detect if there is horizontal symmetry
def hsymmetry(roi):
    if len(roi) % 2 == 0:
        fsthalf = roi[0:len(roi)/2, :]
        sndhalf = roi[len(roi)/2:, :]
        cv2.flip(sndhalf.copy(), 0, sndhalf)
        if fsthalf.__eq__(sndhalf).all():
            return "horizontally symmetrical"
    else:
        fsthalf = roi[0:len(roi)/2, :]
        sndhalf = roi[(len(roi)/2)+1:, :]
        cv2.flip(sndhalf.copy(), 0, sndhalf)
        if fsthalf.__eq__(sndhalf).all():
            return "horizontally symmetrical"
    return None


# Here we detect if there is vertical symmetry
def vsymmetry(roi):
    if len(roi[0]) % 2 == 0:
        fsthalf = roi[:, 0:len(roi[0])/2]
        sndhalf = roi[:, len(roi[0])/2:]
        cv2.flip(sndhalf.copy(), 1, sndhalf)
        if fsthalf.__eq__(sndhalf).all():
            return "vertically symmetrical"
    else:
        fsthalf = roi[:, 0:len(roi[0])/2]
        sndhalf = roi[:, (len(roi[0])/2)+1:]
        cv2.flip(sndhalf.copy(), 1, sndhalf)
        if fsthalf.__eq__(sndhalf).all():
            return "vertically symmetrical"
    return None


# Here we detect whether the shape is "quadrilateral-ish", that is, approaches a rectangle;
# or if it contains substantial negative space in the mbr
def quadrilateral(mbr, shape):
    flag = False
    area_shape = cv2.contourArea(shape)
    # Remove one pixel from each dimension of the bounding rectangle to make it "snug"
    area_mbr = (mbr[0]-1) * (mbr[1]-1)
    # I decided on values of 1.25 and lower for the areas ratio as a good approximation for a rectangular shape
    if (area_mbr / area_shape) <= 1.25:
        flag = True
    return flag


shapes = []
# Here we analyze each shape individually, obtaining basic information such as area and MBR
for shape in contours[0]:
    # Here we get the color (index):
    color = image[shape[0][0][1]][shape[0][0][0]]

    # Here we obtain the minimum bounding rectangle
    x, y, w, h = cv2.boundingRect(shape)

    # We use the MBR to extract the shape as a roi
    roi = image[y:y+h, x:x+w]
    # Sometimes a part of a building gets into the MBR we're working with. This line removes this intrusion:
    roi[roi != color] = 0

    # Does the shape approach a quadrilateral or nah?
    quadr = quadrilateral((w, h), shape)
    if quadr:
        charact.append([color, orientation((w, h))])

    # Is the shape horizontally symmetric? And Vertically?
    hsymm = hsymmetry(roi)
    if hsymm:
        charact.append([color, hsymm])
    vsymm = vsymmetry(roi)
    if vsymm:
        charact.append([color, vsymm])

    # Here we find the area of the shape
    area = int(cv2.contourArea(shape))

    # And here the center of mass
    moments = cv2.moments(shape)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
    center = (cx, cy)

    # Does it look like a letter?
    letter = lettershape(shape)
    if letter:
        charact.append([color, letter])
        charact.append([color, "sharp corners building"])
    corner = corners(shape)
    if corner:
        charact.append([color, corner])

    position = geoposition(center)
    if position:
        charact.append([color, position])

    # All values are added to an array that will be used elsewhere in the program
    shapes.append([color, (x, y, w, h), area, center, quadr, shape])


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

'''
shapes.reverse()
for shape in shapes:
    print names[shape[0]] + " characteristics: "
    print "Center of mass: " + str(shape[3])
    print "Area: " + str(shape[2])
    print "MBR coordinates: " + str(shape[1][0]) + ", " + str(shape[1][1]) + " - " + str(shape[1][0] + shape[1][2]) + ", " + str(shape[1][1] + shape[1][3])
    for item in [p for p in charact if p[0] == shape[0]]:
        print item[1]
    print " "
'''

# Evaluates building spacial relations relations based on angle between each building's center of mass
def spatialrelation(ptS, ptT):
    # Calculate the angle between a vertical line passing through the source and a line from source to target
    a = np.array([ptS[0]+10, ptS[1]])
    b = np.array([ptS[0], ptS[1]])
    c = np.array([ptT[0], ptT[1]])
    ba = a - b
    bc = b - c
    s = np.arctan2(*ba)
    if s < 0:
        s += 2 * np.pi
    e = np.arctan2(*bc)
    if e < 0:
        e += 2 * np.pi
    delta = e - s
    deg = np.rad2deg(delta)
    # I reckon that an angle of 45 to 135 degrees from the first building can be safely considered north, visually
    if 45  < deg <= 135:
        return "N"
    # And so forth for all other cardinal points
    if 135 < deg <= 225:
        return "W"
    if 225 < deg <= 315:
        return "S"
    if deg <= 45 or deg > 315:
        return "E"
    else:
        return False


# Calculates the minimal distance between two shapes
def shapedistance(shapeS, shapeT, area):
    for i in shapeS[::4]:
        min = float("inf")
        for j in shapeT[::4]:
            dx = (i[0][0] - j[0][0])
            dy = (i[0][1] - j[0][1])
            tmpDist = math.hypot(dx, dy)
            if tmpDist < min:
                min = tmpDist
            if tmpDist == 0:
                break  # You can't get a closer distance than 0
    # I believe a threshold of sqrt(area)*3 is as good approximation as any to what constitutes "near" and "far"
    threshold = math.sqrt(area) * 3
    if min < threshold:
        return True
    else:
        return False


relationsN = []
relationsS = []
relationsE = []
relationsW = []
relationsD = []
# Here I evaluate spatial relations between buildings
for shape in shapes:
    for i in shapes:
        # no point comparing a building to itself...
        if shape == i:
            continue
        result = spatialrelation(shape[3], i[3])
        if result == "N":
            relationsN.append([shape[0], i[0], True])
        if result == "S":
            relationsS.append([shape[0], i[0], True])
        if result == "E":
            relationsE.append([shape[0], i[0], True])
        if result == "W":
            relationsW.append([shape[0], i[0], True])
        result = shapedistance(shape[5], i[5], shape[2])
        if result:
            relationsD.append([shape[0], i[0], result])


def transitivefiltering(relations):
    for i in relations:
        for j in relations:
            for k in relations:
                if i[1] == j[0] and i[0] == k[0] and j[1] == k[1] and (i[2] == j[2] == k[2]):
                    relations.pop(relations.index(k))
    return relations

# And now we filter the relations
relationsN = transitivefiltering(relationsN)
relationsS = transitivefiltering(relationsS)
relationsE = transitivefiltering(relationsE)
relationsW = transitivefiltering(relationsW)
relationsD = transitivefiltering(relationsD)

# User interface code
cv2.imshow('campus', display)
frame = np.zeros((495, 700, 3), np.uint8)
frame[:] = (20, 20, 20)

def update(x, y):
    line = 16
    frame[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_PLAIN
    txtcolor = (255, 255, 255)
    index = image[y][x]
    if index == 0:
        # Print only mouse position if the mouse is hovering over the empty spaces
        cv2.putText(frame, 'Mouse pos: ' + str(x) + ", " + str(y), (500, line*30), font, 1, txtcolor, 1, cv2.CV_AA)
    else:
        # Print the building characteristics when the mouse is over one
        cv2.putText(frame, 'Hovering over: ' + names[index], (10, line), font, 1, txtcolor, 1, cv2.CV_AA)
        pos = shapes.index([p for p in shapes if p[0] == index][0])
        ctr = "X: " + str(shapes[pos][3][0]) + ", Y: " + str(shapes[pos][3][1])
        cv2.putText(frame, 'Center of mass: ' + ctr, (10, line*2), font, 1, txtcolor, 1, cv2.CV_AA)
        cv2.putText(frame, 'Area: ' + str(shapes[pos][2]), (10, line*3), font, 1, txtcolor, 1, cv2.CV_AA)
        mbru = "Upper left X: " + str(shapes[pos][1][0]) + " Y: " + str(shapes[pos][1][1])
        mbrl = "Lower right X: " + str(shapes[pos][1][0] + shapes[pos][1][2]) + " Y: " + str(shapes[pos][1][1] + shapes[pos][1][3])
        cv2.putText(frame, 'Minimum bounding rectangle: ', (10, line*4), font, 1, txtcolor, 1, cv2.CV_AA)
        cv2.putText(frame, mbru, (30, line*5), font, 1, txtcolor, 1, cv2.CV_AA)
        cv2.putText(frame, mbrl, (30, line*6), font, 1, txtcolor, 1, cv2.CV_AA)

        cv2.putText(frame, "Building characteristics: ", (10, line*9), font, 1, txtcolor, 1, cv2.CV_AA)
        i = 10
        for item in [p for p in charact if p[0] == index]:
            cv2.putText(frame, item[1], (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
            i += 1
        i += 2

        cv2.putText(frame, "Building relations: ", (10, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
        for item in [p for p in relationsN if p[0] == index]:
            i += 1
            relation = "located north of " + names[item[1]]
            cv2.putText(frame, relation, (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
        for item in [p for p in relationsS if p[0] == index]:
            i += 1
            relation = "located south of " + names[item[1]]
            cv2.putText(frame, relation, (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
        for item in [p for p in relationsE if p[0] == index]:
            i += 1
            relation = "located east of " + names[item[1]]
            cv2.putText(frame, relation, (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
        for item in [p for p in relationsW if p[0] == index]:
            i += 1
            relation = "located west of " + names[item[1]]
            cv2.putText(frame, relation, (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)
        for item in [p for p in relationsD if p[0] == index]:
            i += 1
            relation = "near " + names[item[1]]
            cv2.putText(frame, relation, (30, line*i), font, 1, txtcolor, 1, cv2.CV_AA)

        #cv2.putText(frame, 'Mouse pos: ' + str(x) + ", " + str(y), (500, line*30), font, 1, txtcolor, 1, cv2.CV_AA)
    cv2.imshow("information", frame)


def pointdistance(point, shape):
    min = float("inf")
    for j in shape[::4]:
        dx = (point[0] - j[0][0])
        dy = (point[1] - j[0][1])
        tmpDist = math.hypot(dx, dy)
        if tmpDist < min:
            min = tmpDist
        if tmpDist == 0:
            break  # You can't get a closer distance than 0
    # Here the distance threshold is even more arbitrary. I for now am considering it to be "100 pixels"
    threshold = 100
    if min < threshold:
        return True
    else:
        return False

pixelattrib = []
cloudpixels = []
def computecloud(x, y):
    # Here we compute the descriptions for the point we are studying
    pointattrib = []
    for shape in shapes:
        pointattrib.append([spatialrelation((x, y), shape[3]), pointdistance([x, y], shape[5])])
    if pointattrib == pixelattrib:
        if display[y][x] == 0 or display[y][x] == 255:
            cloudpixels.append([x, y])
            display[y][x] = 128
            if x > 0:
                computecloud(x-1, y)
            if x < len(image[y]) - 1:
                computecloud(x+1, y)
            if y > 0:
                computecloud(x, y-1)
            if y < len(image) - 1:
                computecloud(x, y+1)
        else:
            return
    cv2.imshow('campus', display)
    cv2.imwrite("campus.png", display)


def cloud(x, y):
    global cloudpixels
    cloudpixels = []
    global pixelattrib
    pixelattrib = []
    print "X: " + str(x) + ", Y: " + str(y)
    for shape in shapes:
        result = [spatialrelation((x, y), shape[3]), pointdistance([x, y], shape[5])]
        pixelattrib.append(result)
        if result[0] == "N":
            print "North of " + names[shape[0]]
        if result[0] == "S":
            print "South of " + names[shape[0]]
        if result[0] == "E":
            print "East of " + names[shape[0]]
        if result[0] == "W":
            print "West of " + names[shape[0]]
        if result[1] == True:
            print "Near " + names[shape[0]]
    computecloud(x, y)
    print len(cloudpixels)


def onmouse(event, x, y, flags, param):
    time.sleep(0.01)
    update(x-1, y-1)
    if flags & cv2.EVENT_FLAG_LBUTTON:
        cloud(x, y)

cv2.setMouseCallback("campus", onmouse)
cv2.imshow("information", frame)
cv2.moveWindow("information", 50, 50)
cv2.moveWindow("campus", 765, 50)

cv2.waitKey(0)