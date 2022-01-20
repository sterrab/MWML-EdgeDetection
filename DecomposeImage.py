import cv2
import numpy as np

# ========= READ IMAGE ======================
image = cv2.imread('images/peppers2.tif')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
width = gray_img.shape[1]
height = gray_img.shape[0]

# Display Image
cv2.imshow("Display", gray_img)
cv2.waitKey(0)

# ========= DECOMPOSE IMAGE ======================
from Multiwavelets import *

wavelets_y, wavelets_x, wavelets_xy = calculate_multiwavelet_coeffs(gray_img, width, height)
