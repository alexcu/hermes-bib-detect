#!/usr/bin/env python3
"""
This is a python script to annotate the final output image and produce a final
JSON output file.

Usage:

    python annotate.py /path/to/input/files \
                       /path/to/output \
                       /path/to/text/crops \
                       /path/to/ocr/bboxes \
                       /path/to/aggregate|bib

    If no aggregate dir exists (i.e., NOT cropping on a person), then
    pass in the bib directory for this value instead.

Author: Alex Cummaudo
Date: 23 Aug 2017
"""

import os
import sys
from glob import glob
import cv2
import json
import numpy as np
import re

def annotate_bib_squares(img, bib_bbox, string, bib_accuracy, txt_accuracy):
    """Annotates an image given the bounding box of the bib region.
    Args:
        img (cv2 image): Image read by cv2.
        bib_bbox (dict): Bounding box for the bib region.
        string (string): The string that tesseract read from this bib.
        bib_accuracy (int): The accuracy of bib detection.
        txt_accuracy (int): The accuracy of text detection.
    Returns:
        img (cv2 image): Annotated cv2 image.
    """
    lime = (0,255,0)
    black = (0,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    # Bib regions (draw first)
    x1 = bib_bbox["x1"]
    y1 = bib_bbox["y1"]
    x2 = bib_bbox["x2"]
    y2 = bib_bbox["y2"]
    cv2.rectangle(img, (x1, y1), (x2, y2), lime, 2)
    return img

def annotate_number_labels(img, bib_bbox, string, bib_accuracy, txt_accuracy):
    lime = (0,255,0)
    black = (0,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    # labels for accuracy (overlay)
    x1 = bib_bbox["x1"]
    y1 = bib_bbox["y1"]
    x2 = bib_bbox["x2"]
    y2 = bib_bbox["y2"]
    label = "%s [%s%%/%s%%]" % (string, bib_accuracy, txt_accuracy)
    fnt_sz, baseline = cv2.getTextSize(label, font, 1, 1)
    acc_rect_pt1 = (x1, y1 + baseline - 5)
    acc_rect_pt2 = (x1 + fnt_sz[0] + 5, y1 - fnt_sz[1] - 5)
    cv2.rectangle(img, acc_rect_pt1, acc_rect_pt2, lime, -1)
    cv2.putText(img, label, (x1,y1), font, 1, black)
    return img

def read_json(json_filename):
    """Reads the JSON file as a dictionary.
    Args:
        json_filename (string): The JSON file to read.
    Returns:
        dict: The JSON data, parsed as a dictionary.
    """
    with open(json_filename, 'r') as json_fp:
        json_data = json.load(json_fp)
    return json_data

def main():
    assert len(sys.argv) - 1 >= 5, "Must provide 5 arguments (in_dir, out_dir, text_dir, ocr_dir, aggregate_dir)"

    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"

    text_dir = sys.argv[3]
    assert text_dir != None, "Missing text directory (argv[3])"

    ocr_dir = sys.argv[4]
    assert ocr_dir != None, "Missing string directory (argv[4])"

    aggregate_dir = sys.argv[5]
    assert ocr_dir != None, "Missing aggregate directory (argv[4])"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in glob("%s/*.jpg" % in_dir):
        image_id = os.path.splitext(os.path.basename(file))[0]
        img = cv2.imread(file)
        aggregate_json = read_json("%s/%s.json" % (aggregate_dir, image_id))
        for text_crop_file in glob("%s/%s*.json" % (text_dir, image_id)):
            text_crop_id = os.path.splitext(os.path.basename(text_crop_file))[0]
            # This maps the text crop back to the respective bib...
            matches = re.search(image_id + "_crop_bib_(\d+).json", text_crop_file)
            bib_idx = int(matches.group(1))
            bib_for_text_crop = aggregate_json["bib"]["regions"][bib_idx]
            # Attempt to read string for this text_crop json...
            ocr_bbox_file = "%s/%s_crop_text.json" % (ocr_dir, text_crop_id)
            if not os.path.exists(ocr_bbox_file):
                print("No string file for '%s'. Skipping..." % text_crop_id)
                continue
            # Load the text crops and push them down by the correct origin
            print text_crop_file
            txt_crop_json = read_json(text_crop_file)
            bib_origin_x1 = bib_for_text_crop["x1"]
            bib_origin_y1 = bib_for_text_crop["y1"]
            txt_crop_json["text"]["regions"][0]["x1"] += bib_origin_x1
            txt_crop_json["text"]["regions"][0]["y1"] += bib_origin_y1
            txt_crop_json["text"]["regions"][0]["x2"] += bib_origin_x1
            txt_crop_json["text"]["regions"][0]["y2"] += bib_origin_y1
            # Do the same for each individual character
            ocr_bbox_json = read_json(ocr_bbox_file)
            for ocr in ocr_bbox_json["ocr"]:
                for region in ocr["regions"]:
                    region["x1"] += txt_crop_json["text"]["regions"][0]["x1"]
                    region["y1"] += txt_crop_json["text"]["regions"][0]["y1"]
                    region["x2"] += txt_crop_json["text"]["regions"][0]["x2"]
                    region["y2"] += txt_crop_json["text"]["regions"][0]["y2"]
            # Now annotate the image and JSON
            strings = ','.join([ocr["string"] for ocr in ocr_bbox_json["ocr"]])
            bib_bbox = bib_for_text_crop
            bib_accuracy = int(bib_for_text_crop["accuracy"] * 100)
            txt_accuracy = int(txt_crop_json["text"]["regions"][0]["accuracy"] * 100)
            img = annotate_bib_squares(img, bib_bbox, strings, bib_accuracy, txt_accuracy)
            img = annotate_number_labels(img, bib_bbox, strings, bib_accuracy, txt_accuracy)
            aggregate_json["text"] = txt_crop_json["text"]
            aggregate_json["ocr"] = ocr_bbox_json["ocr"]
        # Now finally spit everything out!
        print aggregate_json
        if "text" not in aggregate_json:
            print("No annotations to be made for '%s' - no text detections. Skipping..." % image_id)
            continue
        out_json_file = ("%s/%s.json" % (out_dir, image_id))
        out_jpeg_file = ("%s/%s.jpg" % (out_dir, image_id))
        print("Writing annotated JSON '%s' to '%s'" % (image_id, out_json_file))
        with open(out_json_file, 'w') as f:
            json.dump(aggregate_json, f)
        print("Writing annotated JPEG '%s' to '%s'" % (image_id, out_jpeg_file))
        cv2.imwrite(out_jpeg_file, img)

if __name__ == '__main__':
    main()
