from goprocam import GoProCamera
from goprocam import constants
import cv2
from time import time
import socket
import numpy as np
import os
from os.path import join
import yaml
# import tensorflow as tf
# import source.libs.align.detect_face as detect_face
# import source.tools as tools
# import sklearn

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam = GoProCamera.GoPro()
gpCam.syncTime()
gpCam.livestream("start")
cap = cv2.VideoCapture("udp://10.5.5.9:8554")
jump_frame = 0

while True:
    start_capture = True
    count = 0
    pre_count = 0
    num_flag = 0
    while True:
        num_flag += 1
        if num_flag%2 == 0:
            continue
        if start_capture:
            nmat, prev_frame = cap.read()
            start_capture = False
            continue
       	nmat, frame = cap.read()
        # cv2.imshow('image',frame)
        if jump_frame != 0:
            jump_frame -= 1
            continue

        pre_count = count
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        thrs = 32
        ret, motion_mask = cv2.threshold(gray_diff, thrs, 1, cv2.THRESH_BINARY)
        flag = np.sum(motion_mask)
        prev_frame = frame.copy()
        print(flag)

        if flag > 60:
            count += 1
        if count == pre_count:
            count = 0

        if count > 3:
            print(str(time()) + ' take_video!')
            gpCam.shoot_video(cfg['live_stream']['video_time'])
            jump_frame = 45

            break

        # cv2.imshow("GoPro OpenCV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time() - t >= 2.5:
            sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
            t=time()
    cv2.destroyAllWindows()


