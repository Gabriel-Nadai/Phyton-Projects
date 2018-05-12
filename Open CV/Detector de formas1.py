import numpy as np
import cv2
import imutils
from time import time
from shapedetector import ShapeDetector

cap = cv2.VideoCapture(0)

# define range of red color in HSV
lower_red = np.array([0, 10, 10])
upper_red = np.array([10, 255, 255])

while (True):
    # Capture frame
    ret, frame = cap.read()

    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # blur and threshold the mask
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    # loop over the contours
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # compute the center of the contour
        M = cv2.moments(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        shape = sd.detect(c)

        if shape == "triangle":
            # draw the contour and center of the shape on the image
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Charging Station", (cX + 10 - int(radius), cY + 25 - int(radius)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, '7337', (530, 450), font, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    cv2.waitKey(10)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()