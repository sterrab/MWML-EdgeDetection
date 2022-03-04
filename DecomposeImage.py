import cv2
import timeit
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

# ========= DECOMPOSE IMAGE - WAVELETS ======================
tic = timeit.default_timer()
from Multiwavelets import *
wavelets_y, wavelets_x, wavelets_xy = calculate_multiwavelet_coeffs(gray_img, width, height)

scaling = {'soft': 1.5, 'extreme': 3}
f_soft = scaling['soft']
f_hard = scaling['extreme']

# Gamma
edge_xy_hard = 0 * wavelets_xy
# edge_xy_soft = 0 * wavelets_xy
q1_xy = np.quantile(wavelets_xy, 0.25)
mean_xy = np.mean(wavelets_xy)
q3_xy = np.quantile(wavelets_xy, 0.75)
lower_bound_xy_hard = q1_xy - f_hard*(q3_xy - q1_xy)
upper_bound_xy_hard = q3_xy + f_hard*(q3_xy - q1_xy)
# lower_bound_xy_soft= q1_xy - f_soft*(q3_xy - q1_xy)
# upper_bound_xy_soft = q3_xy + f_soft*(q3_xy - q1_xy)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_xy[pix_y, pix_x] < lower_bound_xy_hard or wavelets_xy[pix_y, pix_x] > upper_bound_xy_hard:
            edge_xy_hard[pix_y, pix_x] = 255
        # elif wavelets_xy[pix_y, pix_x] < lower_bound_xy_soft or wavelets_xy[pix_y, pix_x] > upper_bound_xy_soft:
        #     edge_xy_soft[pix_y, pix_x] = 255

# edge_xy_diff1 = abs(edge_xy_soft - edge_xy_hard)
# Display Edges - (Soft-Hard)
cv2.imwrite('peppers_edge_xy_f' + str('{:.0f}'.format(10*f_hard)) + '.png', edge_xy_hard)
# cv2.imshow("Edges Gamma - Soft", edge_xy_soft)
# cv2.waitKey(0)
# cv2.imshow("Edges Gamma - Hard", edge_xy_hard)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_xy_diff1.png', edge_xy_diff1)
# cv2.imshow("Edges Gamma - Difference", edge_xy_diff1)
# cv2.waitKey(0)



# Alpha
edge_y_hard = 0 * wavelets_y
# edge_y_soft = 0 * wavelets_y
q1_y = np.quantile(wavelets_y, 0.25)
mean_y = np.mean(wavelets_y)
q3_y = np.quantile(wavelets_y, 0.75)
lower_bound_y_hard = q1_y - f_hard*(q3_y - q1_y)
upper_bound_y_hard = q3_y + f_hard*(q3_y - q1_y)
# lower_bound_y_soft = q1_y - f_soft*(q3_y - q1_y)
# upper_bound_y_soft = q3_y + f_soft*(q3_y - q1_y)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_y[pix_y, pix_x] < lower_bound_y_hard or wavelets_y[pix_y, pix_x] > upper_bound_y_hard:
            edge_y_hard[pix_y, pix_x] = 255
        # elif wavelets_y[pix_y, pix_x] < lower_bound_y_soft or wavelets_y[pix_y, pix_x] > upper_bound_y_soft:
        #     edge_y_soft[pix_y, pix_x] = 255

# edge_y_diff = edge_y_soft - edge_y_hard
# edge_y_diff1 = abs(edge_y_soft - edge_y_hard)
# Display Edges
cv2.imwrite('peppers_edge_y_f' + str('{:.0f}'.format(10*f_hard)) + '.png', edge_y_hard)
# cv2.imshow("Edges Alpha - Soft ", edge_y_soft)
# cv2.waitKey(0)
# cv2.imshow("Edges Alpha - Hard ", edge_y_hard)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_y_diff.png', edge_y_diff)
# cv2.imshow("Edges Alpha - Difference ", edge_y_diff)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_y_diff1.png', edge_y_diff1)
# cv2.imshow("Edges Alpha - Abs Difference ", edge_y_diff1)
# cv2.waitKey(0)

# Beta
edge_x_hard = 0 * wavelets_x
# edge_x_soft = 0 * wavelets_x
q1_x = np.quantile(wavelets_x, 0.25)
mean_x = np.mean(wavelets_x)
q3_x = np.quantile(wavelets_x, 0.75)
lower_bound_x_hard = q1_x - f_hard*(q3_x - q1_x)
upper_bound_x_hard = q3_x + f_hard*(q3_x - q1_x)
# lower_bound_x_soft = q1_x - f_soft*(q3_x - q1_x)
# upper_bound_x_soft = q3_x + f_soft*(q3_x - q1_x)
for pix_x in range(width):
    for pix_y in range(height):
        if wavelets_x[pix_y, pix_x] < lower_bound_x_hard or wavelets_x[pix_y, pix_x] > upper_bound_x_hard:
            edge_x_hard[pix_y, pix_x] = 255
        # elif wavelets_x[pix_y, pix_x] < lower_bound_x_soft or wavelets_x[pix_y, pix_x] > upper_bound_x_soft:
        #     edge_x_soft[pix_y, pix_x] = 255

# edge_x_diff = edge_x_soft - edge_x_hard
# edge_x_diff1 = abs(edge_x_soft - edge_x_hard)
# Display Edges
cv2.imwrite('peppers_edge_x_f' + str('{:.0f}'.format(10*f_hard)) + '.png', edge_x_hard)
# cv2.imshow("Edges Beta - Soft ", edge_x_soft)
# cv2.waitKey(0)
# cv2.imshow("Edges Beta - Hard ", edge_x_hard)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_x_diff.png', edge_x_diff)
# cv2.imshow("Edges Beta - Difference ", edge_x_diff)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_x_diff1.png', edge_x_diff1)
# cv2.imshow("Edges Beta - Abs Difference", edge_x_diff1)
# cv2.waitKey(0)


# # Combo alpha - beta
# edge_x_y_hard = np.clip(edge_x_hard + edge_y_hard, 0, 255)
# edge_x_y_soft = np.clip(edge_x_soft + edge_y_soft, 0, 255)
# # Display Edges
# cv2.imwrite('peppers_edge_x_y_f' + str('{:.0f}'.format(10*f_hard)) + '.png', edge_x_y_hard)
# cv2.imshow("Edges Alpha + Beta - f = 3", edge_x_y_hard)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_x_y_f' + str('{:.0f}'.format(10*f_soft)) + '.png', edge_x_y_soft)
# cv2.imshow("Edges Alpha + Beta - f = 1.5", edge_x_y_soft)
# cv2.waitKey(0)

# Combo alpha - beta - gamma
edge_x_y_xy_hard = np.clip(edge_x_hard + edge_y_hard + edge_xy_hard, 0, 255)
# edge_x_y_xy_soft = np.clip(edge_x_soft + edge_y_soft + edge_xy_soft, 0, 255)
# Display Edges
cv2.imwrite('peppers_edge_x_y_xy_f' + str('{:.0f}'.format(10*f_hard)) + '.png', edge_x_y_xy_hard)
# cv2.imshow("Edges Alpha + Beta + Gamma -  f = 1.5", edge_x_y_xy_hard)
# cv2.waitKey(0)
# cv2.imwrite('peppers_edge_x_y_xy_f' + str('{:.0f}'.format(10*f_soft)) + '.png', edge_x_y_xy_soft)
# cv2.imshow("Edges Alpha + Beta + Gamma -  f = 1.5", edge_x_y_xy_soft)
# cv2.waitKey(0)

toc = timeit.default_timer()
print("Processing Time for Wavelets:", toc-tic)


# ========= DECOMPOSE IMAGE - ANN ======================
tic = timeit.default_timer()
from ANNEdgeDetector import *
model = AdamBCELLlr2
edges_ANN_y, edges_ANN_x = ANNedgedetector(model, gray_img, width, height)
edges_ANN_x_y = np.clip(edges_ANN_y+edges_ANN_x, 0, 255)

cv2.imwrite('peppers_ANN_AdamBCELLlr2_x.png', edges_ANN_x)
cv2.imwrite('peppers_ANN_AdamBCELLlr2_y.png', edges_ANN_y)
cv2.imwrite('peppers_ANN_AdamBCELLlr2_x_y.png', edges_ANN_x_y)

toc = timeit.default_timer()
print("Processing Time for ANN:", toc-tic)




