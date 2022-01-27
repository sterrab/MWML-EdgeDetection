import numpy as np

def calculate_multiwavelet_coeffs(image, width, height):
    # Scaling Coefficients of Internal Cells Only
    scaling_coeffs = 1 / (np.sqrt(width * height)) * image

    vector_H = 1 / np.sqrt(2) * np.ones(2)
    vector_G = 1 / np.sqrt(2) * np.array([1, -1])

    d_alpha = np.zeros((height, width))
    d_beta = np.zeros((height, width))
    d_gamma = np.zeros((height, width))

    for pix_y in range(int(height / 2)):
        for pix_x in range(int(width / 2)):
            # Alpha Modes - Discontinuities in Y
            d_alpha[2 * pix_y, 2*pix_x] = sum(sum(vector_H[itilde] * vector_G[jtilde]
                                                * scaling_coeffs[2 * pix_y + jtilde, 2 * pix_x + itilde]
                                                for itilde in range(2))
                                            for jtilde in range(2))

            d_alpha[2 * pix_y, 2*pix_x + 1] = d_alpha[2 * pix_y, 2*pix_x]

            if pix_y == int(height / 2 - 1):
                d_alpha[2 * pix_y + 1, 2*pix_x] = d_alpha[2 * pix_y, 2*pix_x]
                d_alpha[2 * pix_y + 1, 2 * pix_x +1] = d_alpha[2 * pix_y + 1, 2*pix_x]
            else:
                d_alpha[2 * pix_y + 1, 2*pix_x] = sum(sum(vector_H[itilde] * vector_G[jtilde]
                                                        * scaling_coeffs[2 * pix_y + jtilde + 1,
                                                                         2 * pix_x + itilde]
                                                        for itilde in range(2))
                                                    for jtilde in range(2))

                d_alpha[2 * pix_y + 1, 2*pix_x + 1] = d_alpha[2 * pix_y + 1, 2*pix_x]


            # Beta Modes - Discontinuities in X
            d_beta[2 * pix_y, 2 * pix_x] = sum(sum(vector_G[itilde] * vector_H[jtilde]
                                               * scaling_coeffs[2 * pix_y + jtilde, 2 * pix_x + itilde]
                                               for itilde in range(2))
                                           for jtilde in range(2))

            d_beta[2 * pix_y + 1, 2 * pix_x] = d_beta[2 * pix_y, 2 * pix_x]

            if pix_x == int(width / 2 - 1):
                d_beta[2 * pix_y, 2 * pix_x + 1] = d_beta[2 * pix_y, 2 * pix_x]
                d_beta[2 * pix_y + 1, 2 * pix_x + 1] = d_beta[2 * pix_y, 2 * pix_x + 1]
            else:
                d_beta[2 * pix_y, 2 * pix_x + 1] = sum(sum(vector_G[itilde] * vector_H[jtilde]
                                                       * scaling_coeffs[2 * pix_y + jtilde, 2 * pix_x + itilde + 1]
                                                       for itilde in range(2))
                                                   for jtilde in range(2))

                d_beta[2 * pix_y + 1, 2 * pix_x + 1] = d_beta[2 * pix_y, 2 * pix_x + 1]

            # Gamma Modes - Discontinuities in XY
            d_gamma[2 * pix_y, 2 * pix_x] = sum(sum(vector_G[itilde]
                                                    * vector_G[jtilde]
                                                    * scaling_coeffs[2 * pix_y + jtilde, 2 * pix_x + itilde]
                                                    for itilde in range(2))
                                                for jtilde in range(2))

            if pix_x == int(width / 2 - 1):
                d_gamma[2 * pix_y, 2 * pix_x + 1] = d_gamma[2 * pix_y, 2 * pix_x]
            else:
                d_gamma[2 * pix_y, 2 * pix_x + 1] = sum(sum(vector_G[itilde] * vector_G[jtilde]
                                                            * scaling_coeffs[
                                                                2 * pix_y + jtilde,
                                                                2 * pix_x + itilde + 1]
                                                            for itilde in range(2))
                                                        for jtilde in range(2))

            if pix_y == int(height / 2 - 1):
                d_gamma[2 * pix_y + 1, 2 * pix_x] = d_gamma[2 * pix_y, 2 * pix_x]
            else:
                d_gamma[2 * pix_y + 1, 2 * pix_x] = sum(sum(vector_G[itilde] * vector_G[jtilde]
                                                            * scaling_coeffs[
                                                                2 * pix_y + jtilde + 1,
                                                                2 * pix_x + itilde]
                                                            for itilde in range(2))
                                                        for jtilde in range(2))

            if (pix_x == int(width / 2 - 1)) and (pix_y == int(height / 2 - 1)):
                d_gamma[2 * pix_y + 1, 2 * pix_x + 1] = d_gamma[2 * pix_y, 2 * pix_x]
            elif pix_x == int(width / 2 - 1):
                d_gamma[2 * pix_y + 1, 2 * pix_x + 1] = d_gamma[2 * pix_y + 1, 2 * pix_x]
            elif pix_y == int(height / 2 - 1):
                d_gamma[2 * pix_y + 1, 2 * pix_x + 1] = d_gamma[2 * pix_y, 2 * pix_x + 1]
            else:
                d_gamma[2 * pix_y + 1, 2 * pix_x + 1] = sum(sum(vector_G[itilde] * vector_G[jtilde]
                                                                * scaling_coeffs[
                                                                    2 * pix_y + jtilde + 1,
                                                                    2 * pix_x + itilde + 1]
                                                                for itilde in range(2))
                                                            for jtilde in range(2))



    return d_alpha, d_beta, d_gamma
