#-------------------------------------------------------------------------------
# Name:        Lane Detection
# Purpose:     Program to detect lanes
#
# Author:      poorna
#
# Created:     11-05-2019
# Copyright:   (c) poorna 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])



def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

#Original image is taken and a line is drawn along the LANES
#Image - Original image
#Lines - Hough transformed lines
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #removed unpacking of 1d-list to array as already getting in array
            #form
            print(type(lines))
            cv2.line(line_image, (x1, y1), (x2, y2), (0,0,255), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
#Getting the Canny edge detected image
canny_image = canny(lane_image)
cv2.imshow("Canny edge detected image", canny_image)

#Getting region of interest from canny-edge-detected image
cropped_image = region_of_interest(canny_image)
cv2.imshow("Masked image", cropped_image)

#Applying hough tranform to canny-ROI-cropped_image
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)

line_image = display_lines(lane_image, averaged_lines)
cv2.imshow("Hough transformed image", line_image)

#Multiplies the lane_image with 0.8 - decreasing the intensity or making
#it darker
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)

cv2.imshow("Hough Transformed-Weighted image", combo_image)

#plt.imshow(canny)
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()