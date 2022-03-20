import numpy as np


def compute_order2_modes_from_filtered_image(DG_coeffs, height, width):
    # Initializing modes of up to degree 1
    filtered_modes = np.zeros((2, 2, height, width))

    # Computations from using a 2D Characteristic function as the post-processing SIAC filter
    for pix_y in range(1, int(height - 1)):
        for pix_x in range(1, int(width - 1)):
            # Mode (kx, ky) = (0, 0) - New Average from Filtered Data
            filtered_modes[0, 0, pix_y, pix_x] = 9 / 16 * DG_coeffs[0, 0, pix_y, pix_x] + \
                                                 3 / 32 * (DG_coeffs[0, 0, pix_y - 1, pix_x] +
                                                           DG_coeffs[0, 0, pix_y + 1, pix_x] +
                                                           DG_coeffs[0, 0, pix_y, pix_x - 1] +
                                                           DG_coeffs[0, 0, pix_y, pix_x + 1]) + \
                                                 1 / 64 * (DG_coeffs[0, 0, pix_y - 1, pix_x - 1] +
                                                           DG_coeffs[0, 0, pix_y + 1, pix_x - 1] +
                                                           DG_coeffs[0, 0, pix_y - 1, pix_x + 1] +
                                                           DG_coeffs[0, 0, pix_y + 1, pix_x + 1])
            # Mode (kx, ky) = (1, 0)
            filtered_modes[1, 0, pix_y, pix_x] = 0.10825317547305482457 * (DG_coeffs[0, 0, pix_y, pix_x + 1]
                                                             - DG_coeffs[0, 0, pix_y, pix_x - 1]) + \
                                                 0.018042195912175804096 * (DG_coeffs[0, 0, pix_y - 1, pix_x + 1]
                                                              - DG_coeffs[0, 0, pix_y - 1, pix_x - 1]
                                                              + DG_coeffs[0, 0, pix_y + 1, pix_x + 1]
                                                              - DG_coeffs[0, 0, pix_y + 1, pix_x - 1])
            # Mode (kx, ky) = (0, 1)
            filtered_modes[0, 1, pix_y, pix_x] = 0.10825317547305482457 * (DG_coeffs[0, 0, pix_y + 1, pix_x]
                                                             - DG_coeffs[0, 0, pix_y - 1, pix_x]) + \
                                                 0.018042195912175804096 * (DG_coeffs[0, 0, pix_y + 1, pix_x - 1]
                                                              - DG_coeffs[0, 0, pix_y - 1, pix_x - 1]
                                                              + DG_coeffs[0, 0, pix_y + 1, pix_x + 1]
                                                              - DG_coeffs[0, 0, pix_y - 1, pix_x + 1])
            # Mode (kx, ky) = (1, 1)
            filtered_modes[1, 1, pix_y, pix_x] = 1 / 48 * (DG_coeffs[0, 0, pix_y + 1, pix_x + 1]
                                                           + DG_coeffs[0, 0, pix_y - 1, pix_x - 1]
                                                           - DG_coeffs[0, 0, pix_y + 1, pix_x - 1]
                                                           - DG_coeffs[0, 0, pix_y - 1, pix_x + 1])
    # Boundary pixels
    filtered_modes[0, 0, 0, :] = DG_coeffs[0, 0, 0, :]
    filtered_modes[0, 0, -1, :] = DG_coeffs[0, 0, -1, :]
    filtered_modes[0, 0, :, 0] = DG_coeffs[0, 0, :, 0]
    filtered_modes[0, 0, :, -1] = DG_coeffs[0, 0, :, -1]

    return filtered_modes
