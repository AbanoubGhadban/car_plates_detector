import tensorflow.lite as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pytesseract
import re
import platform

if platform.system().lower() == 'windows':
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

class NumbersExtractor:
    def __init__(self) -> None:
        self.i = 0

    def _get_peaks(self, hist):
        prev = 0
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[prev] < hist[i] and hist[i] > hist[i+1]:
                peaks.append((prev+1+i)//2)
            if hist[i] != hist[i+1]:
                prev = i
        return peaks

    def _get_min_x(self, peaks, hist):
        if len(peaks) < 9:
            return None

        values = list(map(lambda i: hist[i], peaks[:3]))
        std = np.std(values)/np.mean(values)
        l = len(hist)
        if peaks[0]/l >= .03 and std < .05:
            return peaks[2]
        else:
            return peaks[3]

    def get_max_x(self, peaks, hist):
        l = len(hist)
        if (l - peaks[len(peaks) - 1])/l < .03:
            return peaks[len(peaks) - 2]
        return peaks[len(peaks) - 1]

    def get_min_y(self, peaks:list, hist):
        peaks = peaks.copy()
        peaks.sort(key=lambda v: abs((v - len(hist)/2)))
        p1, p2 = peaks[0], peaks[1]
        return p1 if hist[p1] > hist[p2] else p2

    def get_max_y(self, peaks, hist):
        l = len(hist)
        if (l - peaks[len(peaks) - 1])/l < .03:
            return peaks[len(peaks) - 2]
        return peaks[len(peaks) - 1]

    def get_type0_regions(self, img: np.ndarray):
        img_h, img_w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(gray, (5, 5), 0)
        
        h_hist = np.sum(gaus,axis=0).tolist()
        h_peaks = self._get_peaks(h_hist)
        if len(h_peaks) < 9:
            return None
        min_x = self._get_min_x(h_peaks, h_hist)
        max_x = self.get_max_x(h_peaks, h_hist)
        sub_img = gaus[:, min_x:max_x]

        v_hist = np.sum(sub_img, axis=1).tolist()
        v_peaks = self._get_peaks(v_hist)
        if len(v_peaks) < 3:
            return None
        min_y = self.get_min_y(v_peaks, v_hist)
        max_y = self.get_max_y(v_peaks, v_hist)
        first_number = gaus[int(.1*img_h):min_y, min_x:int(min_x + .2*img_w)]
        second_number = gaus[min_y:max_y, min_x:max_x]
        return first_number, second_number

    def plate_to_text(self, img):
        if img is None or img.size == 0:
            return None

        if self.get_plate_type(img) == 1:
            text_region = self.get_type1_region(img)
            if text_region is None or text_region.size == 0:
                return None
            preprocessed = self.preprocess_numbers(text_region)
            data = pytesseract.image_to_string(preprocessed, lang ='eng', config ='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=0123456789')
            return re.sub('[^0-9]', '', data.__str__())
        
        
        top_row, text_region = self.get_type0_regions(img)
        if text_region is None or text_region.size == 0 or top_row is None or top_row.size == 0:
            return None

        preprocessed1 = self.preprocess_numbers(text_region)
        preprocessed2 = self.preprocess_numbers(top_row)
        data1 = pytesseract.image_to_string(preprocessed1, lang ='eng', config ='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=0123456789')
        data2 = pytesseract.image_to_string(preprocessed2, lang ='eng', config ='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=0123456789')
        data1_str = re.sub('[^0-9]', '', data1.__str__())
        data2_str = re.sub('[^0-9]', '', data2.__str__())
        return f'{data2_str}-{data1_str}'

    def preprocess_numbers(self, img):
        print(img.shape)
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_h, t_w = gray.shape
        bordered_img = np.full((t_h + 10, t_w + 10), 255, dtype=np.uint8)
        bordered_img[5:t_h+5, 5:t_w+5] = gray
        return bordered_img

    def get_plate_type(self, img):
        s = img.shape
        img_h, img_w = s[0], s[1]
        if (img_w/img_h) < 1.9:
            return 0
        else:
            return 1

    def get_type1_region(self, img):
        img_h, img_w = img.shape
        sub_img = img[int(0.15*img_h):int(0.89552*img_h), int(0.32*img_w):int(0.95*img_w)]
        return sub_img
