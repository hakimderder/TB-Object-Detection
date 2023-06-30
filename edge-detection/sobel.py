# import sys
import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np

# First we declare the variables we are going to use
window_name = ('Sobel - Edge Detector')
scale = 1
delta = 0
ddepth = cv.CV_16S #cv.CV_16S

# Load the image
cap = cv.VideoCapture(2)


while (1): 
    _, src = cap.read()
    # Check if image is loaded fine
    if src is None:
        print ('Error opening camera')
        break

    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src = cv.GaussianBlur(src, (3, 3), 0)

    # Convert the image to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Gradient-X
    # grad_x = cv.Scharr(gray, ddepth, 1, 0, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) 

    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    ## Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(window_name, grad)

    if cv.waitKey(5) == 27:
        break

cv.destroyAllWindows()
cap.release()
