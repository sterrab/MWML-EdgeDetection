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

edge_xy = 0 * wavelets_xy
q1_xy = np.quantile(wavelets_xy, 0.25)
mean_xy = np.mean(wavelets_xy)
q3_xy = np.quantile(wavelets_xy, 0.75)
lower_bound_xy = q1_xy - 1.5*(q3_xy - q1_xy)
upper_bound_xy = q3_xy + 1.5*(q3_xy - q1_xy)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_xy[pix_y, pix_x] < lower_bound_xy or wavelets_xy[pix_y, pix_x] > upper_bound_xy:
            edge_xy[pix_y, pix_x] = 255

# Display Edges
cv2.imwrite('peppers_edge_xy_f15.png', edge_xy)
cv2.imshow("Edges", edge_xy)
cv2.waitKey(0)


print(wavelets_y)

