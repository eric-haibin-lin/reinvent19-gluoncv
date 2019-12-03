# """Estimate pose from your webcam
# ====================================
# This article will demonstrate how to estimate people's pose from your webcam video stream.

from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints


# Loading the model and webcam
# ----------------------------
# In this tutorial we feed frames from the webcam into a detector, 
# then we estimate the pose for each detected people in the frame.
# For the detector we use ``ssd_512_mobilenet1.0_coco`` as it is fast and accurate enough.

ctx = mx.cpu()
detector = get_model('ssd_512_mobilenet1.0_coco', pretrained=True, ctx=ctx)

# The pre-trained model tries to detect all 80 classes of objects in an image,
# however in pose estimation we are only interested in one object class: person.
# To speed up the detector, we can reset the prediction head to only include the classes we need.

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

# Next for the estimator, we choose ``simple_pose_resnet18_v1b`` for it is light-weighted.

estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

# With OpenCV, we can easily retrieve frames from the webcam.
cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus


# Estimation loop 
# --------------
# For each frame, we perform the following steps:
# - loading the webcam frame
# - pre-process the image
# - detect people in the image
# - post-process the detected people
# - estimate the pose for each person
# - plot the result

axes = None
num_frames = 100

for i in range(num_frames):
    ret, frame = cap.read()
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.2)
    cv_plot_image(img, scale=3)
    cv2.waitKey(1)

# We release the webcam before exiting:
cap.release()
