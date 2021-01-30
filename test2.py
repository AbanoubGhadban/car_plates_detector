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
                    default=r'F:\Work\Car Plate\p_imgs\2.jpg')
                    
args = parser.parse_args()                              

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO IMAGE
IMAGE_PATH = args.image

extractor = NumbersExtractor()
image = cv2.imread(IMAGE_PATH)
rr, r = extractor.get_type0_regions(image)
print(extractor.plate_to_text(image))
cv2.imshow('eee', r)
cv2.waitKey()
