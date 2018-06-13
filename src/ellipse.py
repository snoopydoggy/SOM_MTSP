from math import sqrt, cos, sin, radians

rad90 = radians(360.0)
rad180 = radians(180.0)


def listSum(list, index):
    sum = 0
    for i in list:
        sum = sum + i[index]
    return sum


def calcLength(points):
    prevPoint = points[0]
    for point in points:
        dx = point[0] - prevPoint[0]
        dy = point[1] - prevPoint[1]
        dist = sqrt(dx * dx + dy * dy)
        point[3] = dist
        prevPoint = point


def calcPos(points, a, b):
    angle = 0
    for i in range(1, len(points) - 1):
        point = points[i]
        angle += point[2]
        point[0] = a * cos(angle)
        point[1] = b * sin(angle)


def adjust(points):
    totalLength = listSum(points, 3)
    averageLength = totalLength / (len(points) - 1)

    maxRatio = 0
    for i in range(1, len(points)):
        point = points[i]
        ratio = (averageLength - point[3]) / averageLength
        point[2] = (1.0 + ratio) * point[2]
        absRatio = abs(ratio)
        if absRatio > maxRatio:
            maxRatio = absRatio
    return maxRatio


def ellipse(a, b, steps, limit):
    delta = rad90 / steps

    angle = 0.0

    points = []
    for step in range(steps + 1):
        x = a * cos(angle)
        y = b * sin(angle)
        points.append([x, y, delta, 0.0])
        angle += delta

    doContinue = True
    while doContinue:
        calcLength(points)
        maxRatio = adjust(points)
        calcPos(points, a, b)

        doContinue = maxRatio > limit


    return points
