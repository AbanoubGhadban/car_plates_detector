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
parser.add_argument('--images', help='Directory that contains the test images',
                    default='imgs/')
                    
args = parser.parse_args()                              

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO IMAGE
IMAGES_PATH = args.images

detector = PlateDetector(PATH_TO_MODEL_DIR)
extractor = NumbersExtractor()

for img_path in pathlib.Path(IMAGES_PATH).glob("*.jpg"):
    image = cv2.imread(img_path.__str__())
    image = cv2.resize(image, (480, 480))
    image_cpy = image.copy()
    boxes = detector.get_plates_bounding_boxes(image_cpy)
    for i in range(len(boxes)):
        (minx, miny), (maxx, maxy) = boxes[i]
        t_img, n_img = extractor.get_type0_regions(image_cpy[miny:maxy, minx:maxx])

        text = extractor.plate_to_text(image_cpy[miny:maxy, minx:maxx])
        if n_img is not None and n_img.size > 0 and text is not None:
            cv2.imshow(f'w{i}', n_img)
            preprocessed = extractor.preprocess_numbers(n_img)
            cv2.imshow(f'p{i}', preprocessed)
            cv2.imwrite(f'imgs2\{i}_{img_path.name}', image[miny:maxy, minx:maxx])
        if text is None:
            text = ""
        
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(miny, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (minx, miny), (maxx, maxy), (255, 255, 0), 2)
        cv2.rectangle(image, (minx, label_ymin-labelSize[1]-10), (minx+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, text, (minx, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    
    cv2.imshow(img_path.name, image)
    k = cv2.waitKey()&0xFF
    if k == ord('q'):
        break
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
