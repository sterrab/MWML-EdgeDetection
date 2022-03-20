import numpy as np
import torch

class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H1)
        self.middle_linear = torch.nn.Linear(H1, H2)
        self.output_linear = torch.nn.Linear(H2, D_out)
        self.output_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h_relu1 = self.input_linear(x).clamp(min=0)
        h_relu2 = self.middle_linear(h_relu1).clamp(min=0)
        y_pred = self.output_linear(h_relu2)
        y_pred = self.output_softmax(y_pred)
        return y_pred


def generate_ann_input_data(DG_coeffs, degree, height, width):

    ann_input_data_x = np.zeros((height, width, 5))
    ann_input_data_y = np.zeros((height, width, 5))

    if degree == 0:
        for pix_y in range(2, int(height-2)):
            for pix_x in range(2, int(width-2)):
                # Normalize the 5-pix row stencil input data for a given row pix_y
                row_data = 0.5*DG_coeffs[0, 0, pix_y, pix_x-2:pix_x+3]
                row_data = row_data/max(1, max(abs(row_data)))

                # Normalize the 5-pix col stencil input data for a given col pix_x
                col_data = 0.5*DG_coeffs[0, 0, pix_y-2:pix_y+3, pix_x]
                col_data = col_data / max(1, max(abs(col_data)))

                # Input Data
                ann_input_data_x[pix_y, pix_x, :] = row_data
                ann_input_data_y[pix_y, pix_x, :] = col_data
    elif degree == 1:
        for pix_y in range(1, int(height - 1)):
            for pix_x in range(1, int(width - 1)):
                # Normalize the 3-pix row stencil input data (with left/right reconstructions) for a given row pix_y
                row_data = np.zeros(5)
                for i in range(3):
                    row_data[2*i] = 0.5 * DG_coeffs[0, 0, pix_y, pix_x-1+i]
                    if i == 1:
                        row_data[i] = sum(np.sqrt(0.5)*np.sqrt(kx+0.5)* (-1)**kx
                                          * DG_coeffs[kx, 0, pix_y, pix_x]
                                          for kx in range(degree+1))
                        row_data[i+2] = sum(np.sqrt(0.5) * np.sqrt(kx + 0.5)
                                          * DG_coeffs[kx, 0, pix_y, pix_x]
                                          for kx in range(degree + 1))
                row_data = row_data / max(1, max(abs(row_data)))

                # Normalize the 3-pix col stencil input data (with top/bottom reconstructions) for a given col pix_x
                col_data = np.zeros(5)
                for i in range(3):
                    col_data[2 * i] = 0.5 * DG_coeffs[0, 0, pix_y - 1 + i, pix_x]
                    if i == 1:
                        col_data[i] = sum(np.sqrt(0.5) * np.sqrt(ky + 0.5) * (-1) ** ky
                                          * DG_coeffs[0, ky, pix_y, pix_x]
                                          for ky in range(degree + 1))
                        col_data[i + 2] = sum(np.sqrt(0.5) * np.sqrt(ky + 0.5)
                                              * DG_coeffs[0, ky, pix_y, pix_x]
                                              for ky in range(degree + 1))
                col_data = col_data / max(1, max(abs(col_data)))

                # Input Data
                ann_input_data_x[pix_y, pix_x, :] = row_data
                ann_input_data_y[pix_y, pix_x, :] = col_data
    else:
        print("Current methods consider degree = 0 or degree = 1 only.")

    # Reshape and Transform into tensor
    ann_input_data_x = torch.from_numpy(ann_input_data_x.reshape((height * width, 5)))
    ann_input_data_y = torch.from_numpy(ann_input_data_y.reshape((height * width, 5)))

    return ann_input_data_x, ann_input_data_y

def ann_edge_detector(model_name, H1, H2, DG_coeffs, degree, height, width):

    # Loading Model
    model = ThreeLayerNet(5, H1, H2, 2)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    input_x, input_y = generate_ann_input_data(DG_coeffs, degree, height, width)

    # Evaluate model
    output_y = model(input_y.float())
    output_x = model(input_x.float())

    # Select model classification by finding col index of largest magnitude value
    model_output_y = torch.argmax(output_y, dim=1)
    model_output_x = torch.argmax(output_x, dim=1)

    # Reshape classification flag into image grid
    # -- 1's where the model_output rounds to (0, 1) - Smooth
    # -- 0's where the model_output rounds to (1, 0) - Discontinuous, an Edge
    flag_x = model_output_x.detach().numpy().reshape((height, width))
    flag_y = model_output_y.detach().numpy().reshape((height, width))

    # Converting edges (flag=0) to white=255 in grayscale
    edges_ann_x = abs(255.0 * flag_x - 255)
    edges_ann_y = abs(255.0 * flag_y - 255)

    return edges_ann_y, edges_ann_x

