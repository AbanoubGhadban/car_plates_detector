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

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='model/model.tflite')
parser.add_argument('--image', help='Name of the single image to perform detection on',
                    default='test.jpg')
                    
args = parser.parse_args()                              

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO IMAGE
IMAGE_PATH = args.image

detector = PlateDetector(PATH_TO_MODEL_DIR)
extractor = NumbersExtractor()
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = detector.get_plates_bounding_boxes(image_rgb)
for i in range(len(boxes)):
    (minx, miny), (maxx, maxy) = boxes[i]
    t_img, n_img = extractor.get_type0_regions(image_rgb[miny:maxy, minx:maxx])
    if n_img is not None and n_img.size > 0:
        cv2.imshow(f'w{i}', n_img)
        preprocessed = extractor.preprocess_numbers(n_img)
        cv2.imshow(f'p{i}', preprocessed)
    text = extractor.plate_to_text(image_rgb[miny:maxy, minx:maxx])
    if text is not None:
        print(text)
cv2.waitKey()
