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


def generate_ann_input_data(image, height, width):

    ann_input_data_x = np.zeros((height, width, 5))
    ann_input_data_y = np.zeros((height, width, 5))

    for pix_y in range(2, int(height-2)):
        for pix_x in range(2, int(width-2)):
            # Normalize the 5-col stencil input data for a given row pix_y
            row_data = image[pix_y, pix_x-2:pix_x+3]
            row_data = row_data/max(1, max(abs(row_data)))

            # Normalize the 5-row stencil input data for a given col pix_x
            col_data = image[pix_y-2:pix_y+3, pix_x]
            col_data = col_data / max(1, max(abs(col_data)))

            # Input Data
            ann_input_data_x[pix_y, pix_x, :] = row_data
            ann_input_data_y[pix_y, pix_x, :] = col_data

    # Reshape and Transform into tensor
    ann_input_data_x = torch.from_numpy(ann_input_data_x.reshape((height * width, 5)))
    ann_input_data_y = torch.from_numpy(ann_input_data_y.reshape((height * width, 5)))

    return ann_input_data_x, ann_input_data_y

def ann_edge_detector(model_name, H1, H2, image, height, width):

    # Loading Model
    model = ThreeLayerNet(5, H1, H2, 2)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    input_x, input_y = generate_ann_input_data(image, height, width)

    # Evaluate model
    output_y = model(input_y.float())
    output_x = model(input_x.float())

    # Select model classification by finding col index of largest magnitude value
    model_output_y = torch.argmax(output_y, dim=1)
    model_output_x = torch.argmax(output_x, dim=1)

    # Reshape classification flag into image grid
    # -- 1's where the model_output rounds to (0, 1)
    # -- 0's where the model_output rounds to (1, 0)
    flag_x = model_output_x.detach().numpy().reshape((height, width))
    flag_y = model_output_y.detach().numpy().reshape((height, width))

    # Converting edges to white=255 in grayscale
    edges_ann_x = abs(255.0 * flag_x - 255)
    edges_ann_y = abs(255.0 * flag_y - 255)

    return edges_ann_y, edges_ann_x

def ann_edge_detector_pixbypix(model_name, H1, H2, image, height, width):
    input_x = np.zeros((height, width, 5))
    input_y = np.zeros((height, width, 5))

    output_x_np = np.zeros((height, width, 2))
    output_y_np = np.zeros((height, width, 2))

    edges_ann_x = np.zeros((height, width))
    edges_ann_y = np.zeros((height, width))

    # Loading Model
    model = ThreeLayerNet(5, H1, H2, 2)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    for pix_y in range(2, int(height-2)):
        for pix_x in range(2, int(width-2)):
            # Normalize the 5-col stencil input data for a given row pix_y
            row_data = image[pix_y, pix_x-2:pix_x+3].reshape((1, 5))
            row_data = row_data/max(1, max(abs(row_data[0])))
            input_x[pix_y, pix_x, :] = row_data

            # Normalize the 5-row stencil input data for a given col pix_x
            col_data = image[pix_y-2:pix_y+3, pix_x].reshape((1, 5))
            col_data = col_data / max(1, max(abs(col_data[0])))
            input_y[pix_y, pix_x, :] = col_data

            # Input Data
            ann_input_data_x = torch.from_numpy(row_data)
            ann_input_data_y = torch.from_numpy(col_data)

            # Evaluate model
            output_y = model(ann_input_data_y.float())
            output_x = model(ann_input_data_x.float())

            output_x_np[pix_y, pix_x, :] = output_x.detach().numpy()
            output_y_np[pix_y, pix_x, :] = output_y.detach().numpy()

            # # Select model classification by finding col index of largest magnitude value
            model_output_x = torch.argmax(output_x, dim=1)
            model_output_y = torch.argmax(output_y, dim=1)

            # Edge if col 1 = 1
            if model_output_x == torch.tensor([1]):
                edges_ann_x[pix_y, pix_x] = 255.0

            if model_output_y == torch.tensor([1]):
                edges_ann_y[pix_y, pix_x] = 255.0

    return input_y, input_x, output_y_np, output_x_np, edges_ann_y, edges_ann_x


def compare_input_data(image, height, width):

    input_x, input_y = generate_ann_input_data(image, height, width)

    isEquivalent_x = np.zeros((height, width))
    isEquivalent_y = np.zeros((height, width))

    for pix_y in range(2, int(height-2)):
        for pix_x in range(2, int(width-2)):
            # Normalize the 5-col stencil input data for a given row pix_y
            row_data = image[pix_y, pix_x-2:pix_x+3].reshape((1, 5))
            row_data = row_data/max(1, max(abs(row_data[0])))

            # Normalize the 5-row stencil input data for a given col pix_x
            col_data = image[pix_y-2:pix_y+3, pix_x].reshape((1, 5))
            col_data = col_data / max(1, max(abs(col_data[0])))

            # Input Data
            # ann_input_data_x = torch.from_numpy(row_data)
            # ann_input_data_y = torch.from_numpy(col_data)

            if (input_x[pix_y*width + pix_x, 0] == row_data[0, 0]
                    and input_x[pix_y*width + pix_x, 1] == row_data[0, 1]
                    and input_x[pix_y*width + pix_x, 2] == row_data[0, 2]
                    and input_x[pix_y*width + pix_x, 3] == row_data[0, 3]
                    and input_x[pix_y*width + pix_x, 4] == row_data[0, 4]):
                isEquivalent_x[pix_y, pix_x] = 1

            if (input_y[pix_y*width + pix_x, 0] == col_data[0, 0]
                    and input_y[pix_y*width + pix_x, 1] == col_data[0, 1]
                    and input_y[pix_y*width + pix_x, 2] == col_data[0, 2]
                    and input_y[pix_y*width + pix_x, 3] == col_data[0, 3]
                    and input_y[pix_y*width + pix_x, 4] == col_data[0, 4]):
                isEquivalent_y[pix_y, pix_x] = 1

    return isEquivalent_x, isEquivalent_y
