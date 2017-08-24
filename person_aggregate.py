#!/usr/bin/env python3
"""
This is a python script to aggregate detected bibs from individual person crops
into one image.

Usage:

    python person_aggregate.py /path/to/input/files \
                               /path/to/output \
                               /path/to/person/crops \
                               /path/to/bib/crops

Author: Alex Cummaudo
Date: 23 Aug 2017
"""

import os
import sys
from glob import glob
import cv2
import re
import json
import numpy as np

# Keep unions only if they are 75% of the area of either r1 or r2
KEEP_UNION_THRESHOLD = 0.75

def union(r1, r2):
    """Calculates the union of two regions.
    Args:
        r1, r2 (dict): A dictionary containing {x1, y1, x2, y2} arguments.
    Returns:
        dict: A dictionary in the same fashion.
    """
    x1 = min(r1["x1"], r2["x1"])
    y1 = min(r1["y1"], r2["y1"])
    x2 = max(r1["x2"], r2["x2"])
    y2 = max(r1["y2"], r2["y2"])
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "accuracy": max(r1["accuracy"], r2["accuracy"])
    }

def crop_region(image, region):
    """Crops a singular region in an image
    Args:
        image (image): A numpy image
        region (dict): A dictionary containing x1, y1, x2, y2
    Returns:
        image: The cropped image
    """
    return image[ region["y1"]:region["y2"], region["x1"]:region["x2"] ]

def area(region):
    """Returns the area of the specified region.
    Args:
        region (dict): A dictionary containing {x1, y1, x2, y2} arguments.
    Returns:
        float: The area of the region.
    """
    w = region["x2"] - region["x1"]
    h = region["y2"] - region["y1"]
    return w * h

def intersection(r1, r2):
    """Calculates the intersection rectangle of two regions.
    Args:
        r1, r2 (dict): A dictionary containing {x1, y1, x2, y2} arguments.
    Returns:
        dict or None: A dictionary in the same fashion of just the
                      intersection or None if the regions do not intersect.
    """
    x1 = max(r1["x1"], r2["x1"])
    y1 = max(r1["y1"], r2["y1"])
    x2 = min(r1["x2"], r2["x2"])
    y2 = min(r1["y2"], r2["y2"])
    if y1 < y2 and x1 < x2:
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "accuracy": max(r1["accuracy"], r2["accuracy"])
        }
    else:
        return None

def do_regions_intersect(r1, r2):
    """Calculates whether or not the two regions intersect eachother.
    Args:
        r1, r2 (dict): A dictionary containing {x1, y1, x2, y2} arguments.
    Returns:
        boolean: True if the regions intersect, false otherwise.
    """
    return intersection(r1, r2) is not None

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

def extract_bib_regions(image_filename, bib_json_dir, person_json_dir):
    """Extracts valid bibs from the image.

    By `valid', we mean those regions which do not overlap. We will calculate
    the regions where there are possible overlaps in two people's detections.
    We also adjust these coordinates from the cropped images to the original
    images.

    Args:
        image_filename (string): Path to ORIGINAL image filename.
        bib_json_dir (string): Path of bib JSON files.
        person_json_dir (string): Path of person crop JSON files.
    Returns:
        dict: A mapped dictionary of the same format as Bib JSON but aggregated.
    """
    # Strip the image id from the original filename
    image_id = os.path.splitext(os.path.basename(image_filename))[0]

    # Read in the image
    image = cv2.imread(image_filename)

    # If a person JSON directory is provided, then we need to combine all
    # detected bibs for each respective person
    person_regions = read_json("%s/%s.json" % (person_json_dir, image_id))["person"]["regions"]

    # Now I have all of my person_regions, I cna find the respective bib regions
    # for every single person
    for i, person_region in enumerate(person_regions):
        # These are the person's coordinates in the ORIGINAL image
        px1, py1 = (person_region["x1"], person_region["y1"])
        bib_filename = "%s/%s_crop_person_%i.json" % (bib_json_dir, image_id, i)
        json = read_json(bib_filename)
        person_region["bib_regions"] = json["bib"]["regions"]
        person_region["bib_elapsed_seconds"] = json["bib"]["elapsed_seconds"]

        # Now we must mutate each of these bib regions to be reflective
        # of the ORIGINAL image's dimension sizes
        for bib_region in person_region["bib_regions"]:
            bib_region["x1"] += px1
            bib_region["y1"] += py1
            bib_region["x2"] += px1
            bib_region["y2"] += py1

    # Now strip out all bib regions in the entire photo for every runner
    bib_regions = np.hstack([pr["bib_regions"] for pr in person_regions])
    sum_of_time = np.sum([pr["bib_elapsed_seconds"] for pr in person_regions])

    # Go through every bib region we have, and see if any bibs overlap.
    # If they do, then use the union of both.
    # Go through every bib region we have, and see if any bibs overlap.
    # If they do, then use the union of both.
    bib_regions_to_remove = []
    bib_regions_to_add = []
    for r1 in bib_regions:
        for r2 in bib_regions:
            if r1 == r2:
                continue
            if do_regions_intersect(r1, r2):
                ir = intersection(r1, r2)
                r1a = area(r1)
                r2a = area(r2)
                ira = area(ir)
                # Only if intersection is greater than KEEP_UNION_THRESHOLD
                # If not include this, then too small
                if ira > KEEP_UNION_THRESHOLD * r1a or ira > KEEP_UNION_THRESHOLD * r2a:
                    bib_regions_to_remove.append(r1)
                    bib_regions_to_remove.append(r2)
                    bib_regions_to_add.append(union(r1, r2))
    bib_regions = [r for r in bib_regions if r not in bib_regions_to_remove] + bib_regions_to_add
    # Ensure unique only!!
    bib_regions = np.unique(np.array(bib_regions)).tolist()
    return {
        "bib": { "regions": bib_regions, "elapsed_seconds": sum_of_time }
    }

def crop_bib_regions_from_image(image, bib_regions):
    """Crops the specified bib regions from the given image.
    Args:
        image (string): Path to ORIGINAL image filename.
        bib_regions (dict): The bib regions to crop.
    Returns:
        numpy3d: Numpy 3D array of cropped images
    """

    return [crop_region(image, bib_region) for bib_region in bib_regions]

def aggregate(image_filename, image_id, bib_json_dir, person_json_dir):
    """Aggrates person and bib crops.

    Args:
        image_filename (string): Path to ORIGINAL image filename.
        image_id (string): The identifier of the original image.
        bib_json_dir (string): Path of bib JSON files.
        person_json_dir (string): Path of person crop JSON files.
    """
    person_regions = read_json("%s/%s.json" % (person_json_dir, image_id))
    bib_regions = extract_bib_regions(image_filename, bib_json_dir, person_json_dir)
    return {
        "person": person_regions["person"],
        "bib": bib_regions["bib"]
    }

def main():
    assert len(sys.argv) - 1 >= 4, "Must provide four arguments (in_dir, out_dir, bib_crops_dir, people_dir)"

    in_dir = sys.argv[1]
    assert in_dir != None, "Missing input directory (argv[1])"

    out_dir = sys.argv[2]
    assert out_dir != None, "Missing output directory (argv[2])"

    bib_dir = sys.argv[3]
    assert bib_dir != None, "Missing bib crops directory (argv[3])"

    ppl_dir = sys.argv[4]
    assert ppl_dir != None, "Missing people crops directory (argv[4])"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in glob("%s/*.jpg" % in_dir):
        image_id = os.path.splitext(os.path.basename(file))[0]
        data = aggregate(file, image_id, bib_dir, ppl_dir)
        out_file = ("%s/%s.json" % (out_dir, image_id))
        print("Writing aggregated JSON '%s' to '%s'" % (image_id, out_file))
        with open(out_file, 'w') as f:
            json.dump(data, f)
        image = cv2.imread(file)
        for i, region in enumerate(data["bib"]["regions"]):
            crop = crop_region(image, region)
            crop_file = "%s/%s_crop_bib_%i.jpg"  % (out_dir, image_id, i)
            cv2.imwrite(crop_file, crop)


if __name__ == '__main__':
    main()
