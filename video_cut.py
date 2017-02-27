import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import sys

# -------------------------------------
# Command line argument processing
# -------------------------------------
#if len(sys.argv) < 2:
#    print("Missing image file.")
#    print("python3 lane_detector.py <image_file>")
#
#FILE = str(sys.argv[1])
#
## Read in an test image
#image = mpimg.imread(FILE)


clip = cv2.VideoCapture("project_video.mp4")
#clip = cv2.VideoCapture("harder_challenge_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

frame_cnt = 0
frame_start = 900
frame_end = 0xffffffff
#frame_end = 50

out=None

while True:
    flag, image = clip.read()
    if flag:
        frame_cnt += 1
        if frame_cnt < frame_start:
            continue
        elif frame_cnt > frame_end:
            break
        print('frame_cnt = ', frame_cnt)
        if out == None:
            out = cv2.VideoWriter('frame_gt_900.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))

        cv2.imshow('video', image)
        out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break




