from numbers_extractor import NumbersExtractor
from plate_detector import PlateDetector
import tensorflow.lite as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='model/model.tflite')
                    
args = parser.parse_args()                              

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


detector = PlateDetector(PATH_TO_MODEL_DIR)
extractor = NumbersExtractor()
videostream = VideoStream(resolution=(480,480),framerate=30).start()
time.sleep(1)

while True:
    image = videostream.read()
    image = cv2.resize(image, (480, 480))
    image_cpy = image.copy()
    boxes = detector.get_plates_bounding_boxes(image_cpy)
    for i in range(len(boxes)):
        (minx, miny), (maxx, maxy) = boxes[i]
        text = extractor.plate_to_text(image_cpy[miny:maxy, minx:maxx])
        if text is None:
            text = ""
        
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(miny, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (minx, miny), (maxx, maxy), (255, 255, 0), 2)
        cv2.rectangle(image, (minx, label_ymin-labelSize[1]-10), (minx+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, text, (minx, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    
    cv2.imshow('Webcam', image)
    k = cv2.waitKey(1)&0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()
videostream.stop()
