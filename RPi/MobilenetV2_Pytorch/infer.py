import time

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image

import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name

import os

torch.backends.quantized.engine = 'qnnpack'

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
net = models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)

# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

frame_rate_calc = 1
freq = cv2.getTickFrequency()

classes = label_to_name

with torch.no_grad():
    while True:
    # while cv2.waitKey(1) != ord('q'):
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        
        os.system('clear')

        for idx, val in top[:10]:
            print(f"{val.item()*100:.2f}% {classes(idx)}")

        # print("\r")
        # sys.stdout.write("\033[F")
        # time.sleep(0.5)

        # Our operations on the frame come here
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        # Display the resulting frame
        cv2.imshow('frame', gray)
        
        # Calculate framerate
        # t2 = cv2.getTickCount()
        # time1 = (t2-t1)/freq
        # frame_rate_calc= 1/time1
        
        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:  
            fps = frame_count / (now-last_logged)
            print(f"{fps} fps")
            # cv2.putText(gray,'FPS: {0:.2f}'.format(fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            last_logged = now
            frame_count = 0

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()