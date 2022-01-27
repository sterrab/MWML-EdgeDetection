import cv2
import numpy as np

# ========= READ IMAGE ======================
image = cv2.imread('images/peppers2.tif')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
width = gray_img.shape[1]
height = gray_img.shape[0]

# Display Image
cv2.imwrite('peppers.png', gray_img)
# cv2.imshow("Display", gray_img)
# cv2.waitKey(0)

# ========= DECOMPOSE IMAGE ======================
from Multiwavelets import *

wavelets_y, wavelets_x, wavelets_xy = calculate_multiwavelet_coeffs(gray_img, width, height)

scaling = {'soft': 1.5, 'extreme': 3}
f = scaling['soft']

# Gamma
edge_xy = 0 * wavelets_xy
q1_xy = np.quantile(wavelets_xy, 0.25)
mean_xy = np.mean(wavelets_xy)
q3_xy = np.quantile(wavelets_xy, 0.75)
lower_bound_xy = q1_xy - f*(q3_xy - q1_xy)
upper_bound_xy = q3_xy + f*(q3_xy - q1_xy)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_xy[pix_y, pix_x] < lower_bound_xy or wavelets_xy[pix_y, pix_x] > upper_bound_xy:
            edge_xy[pix_y, pix_x] = 255

# Display Edges
cv2.imwrite('peppers_edge_xy_f' + str(10*f) + '.png', edge_xy)
cv2.imshow("Edges Gamma", edge_xy)
cv2.waitKey(0)

# Alpha
edge_y = 0 * wavelets_y
q1_y = np.quantile(wavelets_y, 0.25)
mean_y = np.mean(wavelets_y)
q3_y = np.quantile(wavelets_y, 0.75)
lower_bound_y = q1_y - f*(q3_y - q1_y)
upper_bound_y = q3_y + f*(q3_y - q1_y)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_y[pix_y, pix_x] < lower_bound_y or wavelets_y[pix_y, pix_x] > upper_bound_y:
            edge_y[pix_y, pix_x] = 255

# Display Edges
cv2.imwrite('peppers_edge_y_f' + str(10*f) + '.png', edge_y)
cv2.imshow("Edges Alpha", edge_y)
cv2.waitKey(0)

# Beta
edge_x = 0 * wavelets_x
q1_x = np.quantile(wavelets_x, 0.25)
mean_x = np.mean(wavelets_x)
q3_x = np.quantile(wavelets_x, 0.75)
lower_bound_x = q1_x - f*(q3_x - q1_x)
upper_bound_x = q3_x + f*(q3_x - q1_x)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_x[pix_y, pix_x] < lower_bound_x or wavelets_x[pix_y, pix_x] > upper_bound_x:
            edge_x[pix_y, pix_x] = 255

# Display Edges
cv2.imwrite('peppers_edge_x_f' + str(10*f) + '.png', edge_x)
cv2.imshow("Edges Beta", edge_x)
cv2.waitKey(0)



print(wavelets_y)

