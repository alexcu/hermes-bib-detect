#!/usr/bin/env python3
"""
This is a python script to annotate the final output image and produce a final
JSON output file. It's called zip because we zip multiple data sources back
together again from the original data source.

Usage:

    python zip.py /path/to/input/files \
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

def annotate_bbox(img, bib_bbox, color=(0,255,0)):
    """Annotates a bbox on an image given the bounding box of the bib region.
    Args:
        img (cv2 image): Image read by cv2.
        bib_bbox (dict): Bounding box for the bib region.
        color (list): Color to annotate.
    Returns:
        img (cv2 image): Annotated cv2 image.
    """
    # Bib regions (draw first)
    x1 = bib_bbox["x1"]
    y1 = bib_bbox["y1"]
    x2 = bib_bbox["x2"]
    y2 = bib_bbox["y2"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def annotate_label(img, bib_bbox, label, color=(0,255,0)):
    """Annotates an bbox's label given the bounding box of the bib region.
    Args:
        img (cv2 image): Image read by cv2.
        bib_bbox (dict): Bounding box for the bib region.
        label (string): The string to label.
        color (list): Color to annotate.
    Returns:
        img (cv2 image): Annotated cv2 image.
    """
    black = (0,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    # labels for accuracy (overlay)
    x1 = bib_bbox["x1"]
    y1 = bib_bbox["y1"]
    x2 = bib_bbox["x2"]
    y2 = bib_bbox["y2"]
    fnt_sz, baseline = cv2.getTextSize(label, font, 1, 1)
    acc_rect_pt1 = (x1, y1 + baseline - 5)
    acc_rect_pt2 = (x1 + fnt_sz[0] + 5, y1 - fnt_sz[1] - 5)
    cv2.rectangle(img, acc_rect_pt1, acc_rect_pt2, color, -1)
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
    assert ocr_dir != None, "Missing ocr directory (argv[4])"

    aggregate_dir = sys.argv[5]
    assert aggregate_dir != None, "Missing aggregate directory (argv[5])"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_json = {}

    for file in glob("%s/*.jpg" % in_dir):
        image_id = os.path.splitext(os.path.basename(file))[0]
        # Explicitly indicate nothing for output file UNLESS overriden at end...
        all_json[image_id] = None
        img = cv2.imread(file)
        aggregate_json_file = "%s/%s.json" % (aggregate_dir, image_id)
        if not os.path.exists(aggregate_json_file):
            print("No aggregate json file for '%s'. Skipping..." % image_id)
            continue
        aggregate_json = read_json(aggregate_json_file)
        aggregate_json["text"] = []
        aggregate_json["ocr"] = []
        for text_crop_file in glob("%s/%s*.json" % (text_dir, image_id)):
            text_crop_id = os.path.splitext(os.path.basename(text_crop_file))[0]
            # This maps the text crop back to the respective bib...
            print("Looking for %s_crop_bib_x.json in %s " % (image_id, text_crop_file))
            matches = re.search(re.escape(image_id) + "_crop_bib_(\d+).json", text_crop_file)
            if matches is None:
                print("No matches found for %s_crop_bib_X.json in %s " % (image_id, text_crop_file))
                continue
            bib_idx = int(matches.group(1))
            bib_for_text_crop = aggregate_json["bib"]["regions"][bib_idx]
            bib_for_text_crop["crop_idx"] = bib_idx
            bib_for_text_crop["is_text_detected"] = True
            # Attempt to read string for this text_crop json...
            ocr_bbox_file = "%s/%s_crop_text.json" % (ocr_dir, text_crop_id)
            if not os.path.exists(ocr_bbox_file):
                print("No string file for '%s'. Skipping..." % text_crop_id)
                bib_for_text_crop["is_text_detected"] = False
                continue
            # Load the text crops and push them down by the correct origin
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
                    region["x2"] += txt_crop_json["text"]["regions"][0]["x1"]
                    region["y2"] += txt_crop_json["text"]["regions"][0]["y1"]
            # Now annotate the image and JSON
            all_strings = [ocr["string"] for ocr in ocr_bbox_json["ocr"]]
            # If there are no string detections (all_strings empty) then
            # we skip this candidate.
            bib_bbox = bib_for_text_crop
            bib_accuracy = int(bib_for_text_crop["accuracy"] * 100)
            txt_accuracy = int(txt_crop_json["text"]["regions"][0]["accuracy"] * 100)
            for ocr in ocr_bbox_json["ocr"]:
                ocr["belongs_to_bib_idx"] = bib_idx
            txt_crop_json["text"]["belongs_to_bib_idx"] = bib_idx
            aggregate_json["text"].append(txt_crop_json["text"])
            aggregate_json["ocr"] = aggregate_json["ocr"] + ocr_bbox_json["ocr"]
            aggregate_json["bib"]["regions"][bib_idx]["rbns"] = all_strings
        # Annotation
        # Annotate each person if exists
        if 'person' in aggregate_json:
            for r in [r for r in aggregate_json['person']['regions']]:
                s = ("Person [c:%s]" % r['accuracy'])
                cyan = (0,255,255)
                img = annotate_bbox(img, r, cyan)
                img = annotate_label(img, r, s, cyan)
        # Annotate each bib region
        for r in [r for r in aggregate_json['bib']['regions']]:
            rbns = ','.join(r['rbns'])
            s = ("Bib [#:%s][c:%s]" % (rbns, r['accuracy']))
            lime = (0,255,0)
            img = annotate_bbox(img, r, lime)
            img = annotate_label(img, r, s, lime)
        # Annotate each text & char region
        all_txt_regions = np.array([txt["regions"] for txt in aggregate_json["text"]]).flatten().tolist()
        all_ocr_regions = np.array([ocr["regions"] for ocr in aggregate_json["ocr"]]).flatten().tolist()
        for r in all_txt_regions:
            white = (255,255,255)
            img = annotate_bbox(img, r, lime)
        for r in all_ocr_regions:
            black = (0,0,0)
            img = annotate_bbox(img, r, lime)
        # Statistics
        all_txt_runtime = np.array([txt["elapsed_seconds"] for txt in aggregate_json["text"]]).flatten().sum()
        all_ocr_runtime = np.array([ocr["elapsed_seconds"] for ocr in aggregate_json["ocr"]]).flatten().sum()
        aggregate_json["stats"] = {
            "num_regions": {
                "bib": len(aggregate_json["bib"]["regions"]),
                "text": len(all_txt_regions),
                "ocr": len(all_ocr_regions)
            },
            "runtime": {
                "bib": aggregate_json["bib"]["elapsed_seconds"],
                "text": all_txt_runtime,
                "ocr": all_ocr_runtime
            }
        }
        # Add person stats if exists
        if "person" in aggregate_json:
            aggregate_json["stats"]["runtime"]["person"] = aggregate_json["person"]["elapsed_seconds"]
            aggregate_json["stats"]["num_regions"]["person"] = len(aggregate_json["person"]["regions"])
        # Now finally spit everything out!
        if len(aggregate_json["text"]) == 0 or len(aggregate_json["ocr"]) == 0:
            print("No annotations to be made for '%s' - no text detections. Skipping..." % image_id)
            continue
        # Indicate all json output should now be the updated aggregate
        all_json[image_id] = aggregate_json
        out_jpeg_file = ("%s/%s.jpg" % (out_dir, image_id))
        print("Writing annotated JPEG '%s' to '%s'" % (image_id, out_jpeg_file))
        cv2.imwrite(out_jpeg_file, img)

    # Writeout global stats file
    out_json_file = ("%s/results.json" % out_dir)
    print("Writing results JSON to '%s'" % out_json_file)
    with open(out_json_file, 'w') as f:
        json.dump(all_json, f)

if __name__ == '__main__':
    main()
