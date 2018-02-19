# import the necessary packages
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread

import argparse
import imutils
import time
import cv2
import os.path

class PiVideoStream:
    def __init__(self, resolution=(2048, 1536), framerate=5):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,\
            format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame
                 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
vs = PiVideoStream().start()
time.sleep(2.0)
criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER + cv2.CALIB_CB_FAST_CHECK

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', (1280,960))

i = 0
while True:
    filename = 'img{}.png'.format(i)
    while os.path.isfile(filename):
        i+=1
        print("Skipping",  filename)
        filename = 'img{}.png'.format(i)
     
    frame = vs.read()
    
    cv2.imshow("Frame", frame)
    # check to see if the frame should be displayed to our screen
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        print('Saving', filename)
        cv2.imwrite(filename, frame)
        i+=1

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
