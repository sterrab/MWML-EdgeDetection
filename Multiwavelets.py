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


def local_outlier_edge_detection(height, width, subdomain_length, type_of_outlier, d_alpha, d_beta, d_gamma):
    # Outlier Detection Parameters and Variables
    scaling = {'soft': 1.5, 'extreme': 3}

    # Initializing Edges Images
    edges_alpha = np.zeros((height, width))
    edges_beta = np.zeros((height, width))
    edges_gamma = np.zeros((height, width))

    # Input Verification and Computing Outlier-based Multiwavelet Edges
    if (type_of_outlier != 'soft') and (type_of_outlier != 'extreme'):
        print('Incorrect Outlier Type. Select "soft" or "extreme".')
    elif ((height % subdomain_length) !=0) or ((width % subdomain_length) !=0):
        print('Select subdomain_length to divide image into uniform local subdomains.')
    else:
        f = scaling[type_of_outlier]

        num_subdomains_x = int(width / subdomain_length)
        num_subdomains_y = int(height / subdomain_length)

        global_mean_alpha = np.mean(abs(d_alpha))
        global_mean_beta = np.mean(abs(d_beta))
        global_mean_gamma = np.mean(abs(d_gamma))

        for subdom_x in range(num_subdomains_x):
            for subdom_y in range(num_subdomains_y):
                # Local Subdomain Boundaries
                x_start = subdom_x * subdomain_length
                x_end = (subdom_x + 1) * subdomain_length
                y_start = subdom_y * subdomain_length
                y_end = (subdom_y + 1) * subdomain_length

                # Boxplot Outlier Fences:
                # ----- Alpha Mode
                q1_alpha = np.quantile(d_alpha[y_start:y_end, x_start:x_end], 0.25)
                q3_alpha = np.quantile(d_alpha[y_start:y_end, x_start:x_end], 0.75)
                lower_bound_alpha = min(-global_mean_alpha, q1_alpha - f * (q3_alpha - q1_alpha))
                upper_bound_alpha = max(global_mean_alpha, q3_alpha + f * (q3_alpha - q1_alpha))

                # ----- Beta Mode
                q1_beta = np.quantile(d_beta[y_start:y_end, x_start:x_end], 0.25)
                q3_beta = np.quantile(d_beta[y_start:y_end, x_start:x_end], 0.75)
                lower_bound_beta = min(-global_mean_beta, q1_beta - f * (q3_beta - q1_beta))
                upper_bound_beta = max(global_mean_beta, q3_beta + f * (q3_beta - q1_beta))

                # ----- Gamma Mode
                q1_gamma = np.quantile(d_gamma[y_start:y_end, x_start:x_end], 0.25)
                q3_gamma = np.quantile(d_gamma[y_start:y_end, x_start:x_end], 0.75)
                lower_bound_gamma = min(-global_mean_gamma, q1_gamma - f * (q3_gamma - q1_gamma))
                upper_bound_gamma = max(global_mean_gamma, q3_gamma + f * (q3_gamma - q1_gamma))

                for pix_x in range(x_start, x_end):
                    for pix_y in range(y_start, y_end):
                        # ----- Alpha Mode
                        if d_alpha[pix_y, pix_x] < lower_bound_alpha or d_alpha[pix_y, pix_x] > upper_bound_alpha:
                            edges_alpha[pix_y, pix_x] = 255

                        # ----- Beta Mode
                        if d_beta[pix_y, pix_x] < lower_bound_beta or d_beta[pix_y, pix_x] > upper_bound_beta:
                            edges_beta[pix_y, pix_x] = 255

                        # ----- Gamma Mode
                        if d_gamma[pix_y, pix_x] < lower_bound_gamma or d_gamma[pix_y, pix_x] > upper_bound_gamma:
                            edges_gamma[pix_y, pix_x] = 255

    return edges_alpha, edges_beta, edges_gamma

def thresholding_edge_detection(height, width, threshold_factor, d_alpha, d_beta, d_gamma):
    # Threshold Values for Each Mode
    threshold_alpha = threshold_factor * np.amax(abs(d_alpha))
    threshold_beta = threshold_factor * np.amax(abs(d_beta))
    threshold_gamma = threshold_factor * np.amax(abs(d_gamma))

    # Initializing Edges Image
    edges_alpha = np.zeros((height, width))
    edges_beta = np.zeros((height, width))
    edges_gamma = np.zeros((height, width))

    for pix_x in range(width):
        for pix_y in range(height):
            # ----- Alpha Mode
            if abs(d_alpha[pix_y, pix_x]) > threshold_alpha:
                edges_alpha[pix_y, pix_x] = 255

            # ----- Beta Mode
            if abs(d_beta[pix_y, pix_x]) > threshold_beta:
                edges_beta[pix_y, pix_x] = 255

            # ----- Gamma Mode
            if abs(d_gamma[pix_y, pix_x]) > threshold_gamma:
                edges_gamma[pix_y, pix_x] = 255

    return edges_alpha, edges_beta, edges_gamma
