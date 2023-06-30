import numpy as np
import cv2 as cv

import time

started = time.time()
last_logged = time.time()
frame_count = 0

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cv.waitKey(1) != ord('q'):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)

    frame_count += 1
    now = time.time()
    if now - last_logged > 1:  
        fps = frame_count / (now-last_logged)
        print(f"{fps} fps")
        cv.putText(frame,'FPS: {0:.2f}'.format(fps),(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv.LINE_AA)
        last_logged = now
        frame_count = 0
    # if cv.waitKey(1) == ord('q'):
        # break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()