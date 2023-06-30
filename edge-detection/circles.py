import sys
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(2)
val = 50

scale = 1
delta = 0
ddepth = cv.CV_16S #cv.CV_16S

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
       
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    gray = cv.GaussianBlur(gray, (3,3), 1.5)
    # gray = cv.medianBlur(gray, 3)

    rows = gray.shape[0]

    # edges = cv.Canny(gray,val,val*3,apertureSize = 3)
    
    # Gradient-X
    # grad_x = cv.Scharr(gray, ddepth, 1, 0, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # grad_x = cv.Sobel(gray, ddepth, 1, 0, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # grad_y = cv.Sobel(gray, ddepth, 0, 1, cv.FILTER_SCHARR, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) 

    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    ## Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8, 
                        param1=100, param2=30,
                        minRadius=100, maxRadius=300)
    # lines = cv.HoughLines(edges,1.2,np.pi/180,200)

    font = cv.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2
    
    if circles is not None:
    
        circles = np.uint16(np.floor(circles))           
        circles2=sorted(circles[0],key=lambda x:x[0],reverse=False)  
    
        print (circles2)
            
        for i in circles2:
            # draw a big point at the center of the circle
            center = (i[0], i[1])
            cv.circle(grad, center, 5, (255, 255, 255), 3)

            # display radius and coordinates
            text = str(i[2])  +' ' +  str(i[0]) +' ' +  str(i[1])
            cv.putText(grad, text, center, font, 1, color, thickness, cv.LINE_AA)
        
    # cv.imshow("detected circles", frame)
    cv.imshow("detected edges", grad)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()