#!/usr/bin/env python3

"""
 Script to preprocess OCR output for Tesseract

 Usage:
 python3 preprocess.py /path/to/input/dir \
                       /path/to/output/dir
"""

from glob import glob
import os
import shutil
import sys
import cv2
import numpy as np

def preprocess(img):
    """Takes a given image and returns the preprocessed version for
    tesseract.

    Args:
        img (cv2 image): The image to preprocess
    Returns
        cv2 image: The preprocessed image.
    """

    # gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray scale inverted
    img_gray_inv = cv2.bitwise_not(img_gray)
    # blurring to increase edges and color contrast:'gaussian convolution' with a kernel
    img_blur = cv2.GaussianBlur(img_gray_inv,ksize=(5,5),sigmaX=0,sigmaY=0)

    # statistical flag for white versus black ID: median
    flag_background = np.median(img_gray)

    # initial idea: 128-ish is half the size of the RGB scale so:
    if flag_background > 128:
        print('White Number Detected!')
        _, threshold = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    else:
        print('Black Number Detected!')
        _, threshold = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # Return the processed image image
    return threshold

def main():
    assert len(sys.argv) - 1 >= 2, "Must provide two arguments (in_dir, out_dir)"

    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in glob("%s/*.jpg" % in_dir):
        print("Processing '%s' for thresholding..." % file)
        img = cv2.imread(file)
        image_id = os.path.splitext(os.path.basename(file))[0]
        out_jpeg_file = ("%s/%s.jpg" % (out_dir, image_id))
        cv2.imwrite(out_jpeg_file, preprocess(img))

    for file in glob("%s/*.json" % in_dir):
        image_id = os.path.splitext(os.path.basename(file))[0]
        out_json_file = ("%s/%s.json" % (out_dir, image_id))
        shutil.copy(file, out_json_file)

if __name__ == '__main__':
    main()
