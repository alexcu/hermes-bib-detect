#!/usr/bin/env python
"""
This is a python script to detect bibs using Keras.
Author: Alex Cummaudo
Date: 1 Aug 2017
"""

from __future__ import division

import pickle
import os
import json

from time import time as now
from optparse import OptionParser
from keras_frcnn import config
from keras_frcnn import roi_helpers
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
import numpy as np

# Parser
OPTS_PARSER = OptionParser()
OPTS_PARSER.add_option("-i", dest="input_file", help="File to process")
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
    if not options.input_file:
        OPTS_PARSER.error("Missing input file")
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

def bboxes_to_dict(bboxes, new_probs, ratio):
    """Converts the bounding boxes to per-pixel coords as a dictionary.
    Args:
        bboxes (list): A list of all the bounding boxes detected.
        new_probs (list): A list containing the estimated accuracy.
        ratio (float): The aspect ratio to convert.
    Returns:
        list: A mapped list to per-pixel coordinates as a dictionary.
    """
    result_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(
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
            accuracy = new_probs[jk]
            result_dets.append({
                "x1": real_x1,
                "y1": real_y1,
                "x2": real_x2,
                "y2": real_y2,
                "accuracy": accuracy
            })
    return result_dets

def annotate_image(image, detections):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.set_axis_off()
    for region in detections:
        x1 = region["x1"]
        y1 = region["y1"]
        x2 = region["x2"]
        y2 = region["y2"]
        acc = region["accuracy"]
        w = x2 - x1
        h = y2 - y1
        ax.add_patch(
            patches.Rectangle(
                (x1,y1),
                width=w,
                height=h,
                linewidth=6,
                fill=False,
                color="lime"
            )
        )
        ax.text(
            x1+10,
            y1-30,
            ("%s%%" % acc * 100),
            backgroundcolor="lime",
            color="black",
            size=15
        )
    return fig

def process_image(image_filename, config, models):
    """Process the given image using FRCNN.
    Args:
        image_filename (string): The image to process.
        config (object): Configuration settings.
        models (tuple): The models generated from Keras.
    Returns:
        list or None: Returns a list of detected bounding boxes, mapped to
                      their per-pixel coordinates, or none if failed.
    """
    print("Processing: %s..." % image_filename)

    if not os.path.exists(image_filename):
        print("No such image at %s. Aborting." % image_filename)
        return None

    img = cv2.imread(image_filename)
    x, ratio = format_image(img, config)
    if K.image_dim_ordering() == 'tf':
        x = np.transpose(x, (0, 2, 3, 1))

    # Unpack the models
    model_rpn, model_classifier_only, model_class = models

    # get the feature maps and output from the RPN
    [y1, y2, F] = model_rpn.predict(X)
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

    # FRCNN algo
    for jk in range(R.shape[0] // config.num_rois + 1):
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

        [p_cls, p_regr] = model_classifier_only.predict([F, rois])

        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(
                    p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_name = config.class_mapping[np.argmax(p_cls[0, ii, :])]

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
    return bboxes_to_dict(bboxes, new_probs, ratio)

def main():
    """Main program entry point"""
    options = OPTS_PARSER.parse_args()[0]

    if not check_options(options):
        return

    # Load config
    config = load_config(options.config_file)

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    start_time = now()
    models = configure_keras_models(config)
    detections = process_image(options.input_file, config, models)

    if detections != None:
        for region in detections:
            x1 = region["x1"]
            y1 = region["y1"]
            x2 = region["x2"]
            y2 = region["y2"]
            acc = region["accuracy"]
            w = x2 - x1
            h = y2 - y1
            print("Detection at (%s, %s) to (%s, %s) "
            "(w=%s, h=%s) "
            "[Accuracy = %s%%]" % (x1, y1, x2, y2, w, h, acc * 100))

    elapsed_time = start_time - now()
    print("Time taken: %sms." % elapsed_time)

    input_file_basename = os.path.basename(options.input_file)

    # Writing out annotated image done if not doing json only
    if not config.json_only:
        print("Annotating image...")
        img_annotated = annotate_image(image, detections)
        img_annotated_file = "%s/%s" % (options.output_dir, input_file_basename)
        print("Writing annotated image to '%s'..." % img_annotated_file)
        mpimg.imsave(img_annotated)

    # JSON processing done if not doing image only
    if not config.image_only:
        data = {"bib_regions": detections, "elapsed_time": elapsed_time}
        json_filename = "%s/%s.json" % (
            options.output_dir,
            os.path.splitext(input_file_basename)[0]
        )
        print("Writing JSON file to '%s'..." % json_filename)
        with open(json_filename) as outfile:
             json.dump(data, outfile)


# Start of script
if __name__ == '__main__':
    main()

