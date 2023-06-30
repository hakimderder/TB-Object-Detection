import numpy as np
import cv2 as cv
import argparse
import imutils

cap = cv.VideoCapture(2)

while True:
    _, image = cap.read()
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    c = max(cnts, key=cv.contourArea)

    # output = image
    cv.drawContours(image, c, -1, (0, 0, 255), 3)
    (x, y, w, h) = cv.boundingRect(c)
    text = "orignal, num_pts=()".format(len(c))
    cv.putText(image, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # epsilon = 0.1*cv.arcLength(cnt, True)
    # approx = cv.approxPolyDP(cnt, epsilon, True)
    
    # cv.drawContours(img, approx, -1, (0, 0, 255), 3)
    # cv.imshow("Contours", img)

    # to demonstrate the impact of contour approximation, let's loop
    # over a number of epsilon sizes
    for eps in np.linspace(0.001, 0.05, 10):
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, eps * peri, True)
        # draw the approximated contour on the image
        # output = image.copy()
        cv.drawContours(image, [approx], -1, (0, 255, 0), 3)
        text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
        cv.putText(image, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2)
        # show the approximated contour image
        print("[INFO] {}".format(text))
        cv.imshow("Approximated Contour", image)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()