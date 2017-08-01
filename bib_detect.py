#!/usr/bin/env python
"""
This is a python script to detect bibs using Keras.
Author: Alex Cummaudo
Date: 1 Aug 2017
"""

from __future__ import division

import pickle
import os
from glob import glob
import json

from time import time as now
from optparse import OptionParser
from keras_frcnn import config
from keras_frcnn import roi_helpers
from keras import backend as K
from keras.layers import Input
from keras.models import Model

import cv2
import numpy as np

BOUNDING_BOX_THRESH = 0.8

# Parser
OPTS_PARSER = OptionParser()
OPTS_PARSER.add_option("-i", dest="input_dir", help="Input directory to process")
OPTS_PARSER.add_option("-o", dest="output_dir", help="Directory to put output")
OPTS_PARSER.add_option("-j", dest="json_only", help="Output JSON only", default=False)
OPTS_PARSER.add_option("-g", dest="image_only", help="Output image only", default=False)
OPTS_PARSER.add_option("-c", dest="config_file", help="Pickle config file")

def check_options(options):
    """Checks if the provided options are ok.

    Args:
        options (object): The provided options.
    Returns:
        boolean: True if ok, false otherwise.
    """
    if not options.input_dir:
        OPTS_PARSER.error("Missing input directory")
        return false
    if not options.output_dir:
        OPTS_PARSER.error("Missing output directory")
        return False
    if not options.config_file:
        OPTS_PARSER.error("Missing config file")
        return False
    return True

def load_config(filename):
    """Loads in configuration file.

    Args:
        filename (string): The filename of the config file.
    Returns:
        object: A deserialised pickle object.
    """
    with open(filename, 'rb') as file_ptr:
        return pickle.load(file_ptr)

def format_image(img, config):
    """Formats the image size based on config.

    Args:
        img (3D numpy array): The image to process.
        config (object): Configuration object.

    Returns:
        tuple: Containing the resized image & channels with the new ratio
    """
    # Image resize
    img_min_side = float(config.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)

    img = cv2.resize(img, (new_width, new_height),
                     interpolation=cv2.INTER_CUBIC)

    # Channel modifications
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= config.img_channel_mean[0]
    img[:, :, 1] -= config.img_channel_mean[1]
    img[:, :, 2] -= config.img_channel_mean[2]
    img /= config.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return (img, ratio)

def configure_keras_models(config):
    """Configures Keras.

    Args:
        nn (Neural Net): A keras NN.
        config (object): The config file.
    Returns:
        tuple: A tuple of three classifiers: (1) the rpn classifier, (2) the
               classifier (only), (3) the classifier.
    """
    # Import the correct NN according to config
    # This must be called within main so that import is called
    if config.network == 'resnet50':
        import keras_frcnn.resnet as nn
        num_features = 1024
    elif config.network == 'vgg':
        import keras_frcnn.vgg as nn
        num_features = 512

    # Configure Keras
    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

    # Config inputs
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(config.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # Classifier to return
    classifier = nn.classifier(
        feature_map_input,
        roi_input,
        config.num_rois,
        nb_classes=len(config.class_mapping),
        trainable=True
    )

    print("Configuring Keras with:")
    print("  - Neural Network: %s..." % config.network)
    print("  - Weights loaded from: %s" % config.model_path)
    print("  - Dimension Ordering: %s" % K.image_dim_ordering())
    print("  - Num Features: %s" % num_features)
    print("  - Num rois: %s" % config.num_rois)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(config.model_path, by_name=True)
    model_classifier.load_weights(config.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    return (model_rpn, model_classifier_only, model_classifier)

def bboxes_to_dict(bboxes, probs, ratio):
    """Converts the bounding boxes to per-pixel coords as a dictionary.
    Args:
        bboxes (list): A list of all the bounding boxes detected.
        probs (list): A list containing the estimated accuracy.
        ratio (float): The aspect ratio to convert.
    Returns:
        list: A mapped list to per-pixel coordinates as a dictionary.
    """
    result_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, probs = roi_helpers.non_max_suppression_fast(
            bbox,
            np.array(probs[key]),
            overlap_thresh=0.5
        )
        for jk in range(new_boxes.shape[0]):
            x1, y1, x2, y2 = new_boxes[jk, :]
            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))
            accuracy = float(probs[jk])
            result_dets.append({
                "x1": real_x1,
                "y1": real_y1,
                "x2": real_x2,
                "y2": real_y2,
                "accuracy": accuracy
            })
    return result_dets

def annotate_image(img, detections):
    lime = (0,255,0)
    black = (0,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    # Bib regions (draw first)
    for region in detections:
        x1 = region["x1"]
        y1 = region["y1"]
        x2 = region["x2"]
        y2 = region["y2"]
        cv2.rectangle(img, (x1, y1), (x2, y2), lime, 2)
    # labels for accuracy (overlay)
    for region in detections:
        x1 = region["x1"]
        y1 = region["y1"]
        x2 = region["x2"]
        y2 = region["y2"]
        acc = region["accuracy"]
        label = "%s%%" % int(acc * 100)
        fnt_sz, baseline = cv2.getTextSize(label, font, 1, 1)
        acc_rect_pt1 = (x1, y1 + baseline - 5)
        acc_rect_pt2 = (x1 + fnt_sz[0] + 5, y1 - fnt_sz[1] - 5)
        cv2.rectangle(img, acc_rect_pt1, acc_rect_pt2, lime, -1)
        cv2.putText(img, label, (x1,y1), font, 1, black)
    return img

def validate_image(img, config, models):
    """Validates the given image using FRCNN.
    Args:
        img (numpy 3D array): The image to process.
        config (object): Configuration settings.
        models (tuple): The models generated from Keras.
    Returns:
        list: Returns a list of detected bounding boxes, mapped to
              their per-pixel coordinates.
    """
    x, ratio = format_image(img, config)
    if K.image_dim_ordering() == 'tf':
        x = np.transpose(x, (0, 2, 3, 1))

    # Unpack the models
    model_rpn, model_classifier_only, model_class = models

    # get the feature maps and output from the RPN
    [y1, y2, f] = model_rpn.predict(x)
    roi = roi_helpers.rpn_to_roi(
        y1,
        y2,
        config,
        K.image_dim_ordering(),
        overlap_thresh=0.7
    )

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    roi[:, 2] -= roi[:, 0]
    roi[:, 3] -= roi[:, 1]

    # Convert classes to a dict
    class_mapping = { v: k for k, v in config.class_mapping.items() }

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    # FRCNN algo
    for jk in range(roi.shape[0] // config.num_rois + 1):
        rois = np.expand_dims(
            roi[config.num_rois * jk:config.num_rois * (jk + 1), :],
            axis=0
        )
        if rois.shape[1] == 0:
            break

        if jk == roi.shape[0] // config.num_rois:
            curr_shape = rois.shape
            target_shape = (curr_shape[0], config.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        [p_cls, p_regr] = model_classifier_only.predict([f, rois])

        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < BOUNDING_BOX_THRESH or np.argmax(
                    p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(p_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = rois[0, ii, :]

            cls_num = np.argmax(p_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([
                config.rpn_stride * x,
                config.rpn_stride * y,
                config.rpn_stride * (x + w),
                config.rpn_stride * (y + h)
            ])
            probs[cls_name].append(np.max(p_cls[0, ii, :]))

    # Convert all bboxes to list of dict items
    return bboxes_to_dict(bboxes, probs, ratio)

def process_image(image_filename, config, options):
    """Processes validation on the given image.
    Args:
        image_filename (string): The image to process.
        config (object): Configuration settings.
        options (object): Parsed command options.
    """
    print("Processing: %s..." % image_filename)

    if not os.path.exists(image_filename):
        print("No such image at %s. Skipping." % image_filename)
        return

    img = cv2.imread(image_filename)
    if img == None:
        print("Cannot read this image properly. Skipping.")
        return

    start_time = now()
    models = configure_keras_models(config)
    detections = validate_image(img, config, models)

    if detections != None:
        for region in detections:
            x1 = region["x1"]
            y1 = region["y1"]
            x2 = region["x2"]
            y2 = region["y2"]
            acc = region["accuracy"]
            w = x2 - x1
            h = y2 - y1
            print("!! Detection at (%s, %s) to (%s, %s) "
            "(w=%s, h=%s) "
            "[Accuracy = %s%%]" % (x1, y1, x2, y2, w, h, int(acc * 100)))

    elapsed_time = now() - start_time
    print("Time taken: %ss." % elapsed_time)

    input_file_basename = os.path.basename(image_filename)

    # Writing out annotated image done if not doing json only
    if not options.json_only:
        print("Annotating image...")
        img_annotated = annotate_image(img, detections)
        img_annotated_file = "%s/%s" % (options.output_dir, input_file_basename)
        print("Writing annotated image to '%s'..." % img_annotated_file)
        cv2.imwrite(img_annotated_file, img_annotated)

    # JSON processing done if not doing image only
    if not options.image_only:
        data = {"bib_regions": detections, "elapsed_time": elapsed_time}
        json_filename = "%s/%s.json" % (
            options.output_dir,
            os.path.splitext(input_file_basename)[0]
        )
        print("Writing JSON file to '%s'..." % json_filename)
        with open(json_filename, 'w') as outfile:
             json.dump(data, outfile)

def main():
    """Main program entry point"""
    options = OPTS_PARSER.parse_args()[0]

    if not check_options(options):
        return

    # Load config
    config = load_config(options.config_file)

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    # Process every image
    for image in glob("%s/*.jpg" % options.input_dir):
        process_image(image, config, options)

# Start of script
if __name__ == '__main__':
    main()

