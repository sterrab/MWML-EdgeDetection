import cv2
import timeit
import os
import numpy as np

# Function to CONVERT EDGES to Black
def convert_edges_to_black(edge_image):
    return abs(edge_image - 255)

# ========= SELECT IMAGE to be Edge-Detected ================
# image_name = 'peppers'
image_name = 'SheppLogan'
image = cv2.imread('images/' + image_name + '.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
width = gray_img.shape[1]
height = gray_img.shape[0]


# ========= SELECT DEGREE for Image Data ====================
# Select degree of Image/DG/ANN data:
# degree = 0 => given data || degree = 1 => use filtered data
degree = 0

# Create Directory for Edge Images
directory = os.getcwd()
image_dir = 'edge images degree ' + str(degree)
if not os.path.exists(directory + '/' + image_dir):
    os.makedirs(directory + '/' + image_dir)

# ========== COMPUTE DG DATA =================================

tic1 = timeit.default_timer()
# 1) -------- Get DG Coefficient Data -------------------

# The pixel average/value is half times the (0,0) mode
# DG_coeffs computation = 2 x pixel data, reshaped to be in shape (degree_x, degree_y, height, width)
DG_coeffs = np.zeros((1, 1, height, width))
DG_coeffs[0, 0, :, :] = 2.0 * gray_img

if degree == 1:
    from FilteredImageData import *
    DG_coeffs = compute_order2_modes_from_filtered_image(DG_coeffs, height, width)
elif degree > 1:
    print("Current methods consider degree = 0 or degree = 1 only.")

toc1 = timeit.default_timer()

# # ========= DETECT EDGES using MULTIWAVELETS =================

# 1) -------- Compute Multiwavelets -------------------
from Multiwavelets import *

tic2 = timeit.default_timer()
wavelets_y, wavelets_x, wavelets_xy = calculate_multiwavelet_coeffs(DG_coeffs, degree, width, height)

toc2 = timeit.default_timer()

# 2) -------- Apply Either Local Outlier Detection or Thresholding -------------------
### Local Outlier Detection Method
tic3 = timeit.default_timer()
subdomain_dimension = width
outlier_type = 'extreme'
edges_lod_y, edges_lod_x, edges_lod_xy = local_outlier_edge_detection(height, width, subdomain_dimension, outlier_type,
                                                                      wavelets_y, wavelets_x, wavelets_xy)
edges_lod_x_y = np.clip(edges_lod_y + edges_lod_x, 0, 255)
edges_lod_x_y_xy = np.clip(edges_lod_y + edges_lod_x + edges_lod_xy, 0, 255)

# ----Convert Edges to Black, Non-Edges to White
edges_lod_y = convert_edges_to_black(edges_lod_y)
edges_lod_x = convert_edges_to_black(edges_lod_x)
edges_lod_xy = convert_edges_to_black(edges_lod_xy)
edges_lod_x_y = convert_edges_to_black(edges_lod_x_y)
edges_lod_x_y_xy = convert_edges_to_black(edges_lod_x_y_xy)

# ----Save Edge Images


image_path = directory + '/' + image_dir + '/' + image_name + 'LOD' + outlier_type + str(subdomain_dimension)
cv2.imwrite(image_path + '_y.png', edges_lod_y)
cv2.imwrite(image_path + '_x.png', edges_lod_x)
cv2.imwrite(image_path + '_xy.png', edges_lod_xy)
cv2.imwrite(image_path + '_x_y.png', edges_lod_x_y)
cv2.imwrite(image_path + '_x_y_xy.png', edges_lod_x_y_xy)

# ----Display Images (UnComment Below)
# cv2.imshow('Wavelet-based Local Outlier Detection Edges: Y direction', edges_lod_y)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Local Outlier Detection Edges: X direction', edges_lod_x)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Local Outlier Detection Edges: XY direction', edges_lod_xy)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Local Outlier Detection Edges: X+Y directions', edges_lod_x_y)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Local Outlier Detection Edges: X+Y+XY directions', edges_lod_x_y_xy)
# cv2.waitKey(0)

toc3 = timeit.default_timer()
print("Processing Time for Wavelets-based Local Outlier Detection Wavelets:", toc1-tic1 + toc2-tic2 + toc3-tic3)

### Thresholding Method
tic4 = timeit.default_timer()
threshold_factor = 0.05
edges_thresh_y, edges_thresh_x, edges_thresh_xy = thresholding_edge_detection(height, width, threshold_factor,
                                                                      wavelets_y, wavelets_x, wavelets_xy)
edges_thresh_x_y = np.clip(edges_thresh_y + edges_thresh_x, 0, 255)
edges_thresh_x_y_xy = np.clip(edges_thresh_y + edges_thresh_x + edges_thresh_xy, 0, 255)

# ----Convert Edges to Black, Non-Edges to White
edges_thresh_y = convert_edges_to_black(edges_thresh_y)
edges_thresh_x = convert_edges_to_black(edges_thresh_x)
edges_thresh_xy = convert_edges_to_black(edges_thresh_xy)
edges_thresh_x_y = convert_edges_to_black(edges_thresh_x_y)
edges_thresh_x_y_xy = convert_edges_to_black(edges_thresh_x_y_xy)

# ----Save Edge Images
image_path = directory + '/' + image_dir+ '/' + image_name + 'Thresh' + str(threshold_factor)
cv2.imwrite(image_path + '_y.png', edges_thresh_y)
cv2.imwrite(image_path + '_x.png', edges_thresh_x)
cv2.imwrite(image_path + '_xy.png', edges_thresh_xy)
cv2.imwrite(image_path + '_x_y.png', edges_thresh_x_y)
cv2.imwrite(image_path + '_x_y_xy.png', edges_thresh_x_y_xy)

# ----Display Images (UnComment Below)
# cv2.imshow('Wavelet-based Threshold Edges: Y direction', edges_thresh_y)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Threshold Edges: X direction', edges_thresh_x)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Threshold Edges: XY direction', edges_thresh_xy)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Threshold Edges: X+Y directions', edges_thresh_x_y)
# cv2.waitKey(0)
# cv2.imshow('Wavelet-based Threshold Edges: X+Y+XY directions', edges_thresh_x_y_xy)
# cv2.waitKey(0)

toc4 = timeit.default_timer()
print("Processing Time for Thresholded Wavelets:", toc1-tic1 + toc2-tic2 + toc4-tic4)

# ========= DETECT EDGES using an ANN MODEL  ======================
tic5 = timeit.default_timer()
from ANNEdgeDetector import *

if degree == 0:
    # Input Data: 5-cell stencil of pixel averages
    model = 'Adamlr1e-4BCELL.pt'
    H1 = 10
    H2 = 10
else:
    # Input Data: 3-cell stencil of pixel averages and middle cell edge reconstructions
    model = '3cellStencil84SGDm09lr1e-4BCE.pt'
    H1 = 8
    H2 = 4

edges_ann_y, edges_ann_x = ann_edge_detector(directory + '/' + model, H1, H2, DG_coeffs, degree, height, width)
edges_ann_x_y = np.clip(edges_ann_y + edges_ann_x, 0, 255)

# ----Convert Edges to Black, Non-Edges to White
edges_ann_y = convert_edges_to_black(edges_ann_y)
edges_ann_x = convert_edges_to_black(edges_ann_x)
edges_ann_x_y = convert_edges_to_black(edges_ann_x_y)

# ----Save Edge Images
image_path = directory + '/' + image_dir + '/' + image_name + 'ANN' + str(model)
cv2.imwrite(image_path + '_y.png', edges_ann_y)
cv2.imwrite(image_path + '_x.png', edges_ann_x)
cv2.imwrite(image_path + '_x_y.png', edges_ann_x_y)

# ----Display Images (UnComment Below)
# cv2.imshow('ANN-based Edges: Y direction', edges_ann_y)
# cv2.waitKey(0)
# cv2.imshow('ANN-based Edges: X direction', edges_ann_x)
# cv2.waitKey(0)
# cv2.imshow('ANN-based Edges: X+Y directions', edges_ann_x_y)
# cv2.waitKey(0)

toc5 = timeit.default_timer()
print("Processing Time for ANN Edges:", toc1-tic1 + toc5 - tic5)

