import numpy as np
import torch
import cv2
import os

from ANNEdgeDetector import *

# ============ TEST IMAGE ==============
image = cv2.imread('images/SheppLogan.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
width = gray_img.shape[1]
height = gray_img.shape[0]


# ============ RUNNING ALL ANN MODELS ================
directory = '8x4'
image_dir = 'edge images'
if not os.path.exists(directory + '/' + image_dir):
    os.makedirs(directory + '/' + image_dir)

# Model Parameters
H1 = 8
H2 = 4

# List of all model files
models = [file for file in os.listdir(directory) if file[-3:] == ".pt"]

for model in models:
    model_path = directory + '/' + model
    print(model_path)
    # ANN Edge Detector Evaluation
    edges_ANN_y, edges_ANN_x = ann_edge_detector(model_path, H1, H2, gray_img, height, width)
    edges_ANN_x_y = np.clip(edges_ANN_y + edges_ANN_x, 0, 255)

    # Saving Images
    image_path = directory + '/' + image_dir + '/' + model[:-3]
    cv2.imwrite(image_path + '_SL_x.png', edges_ANN_x)
    cv2.imwrite(image_path + '_SL_y.png', edges_ANN_y)
    cv2.imwrite(image_path + '_SL_x_y.png', edges_ANN_x_y)


